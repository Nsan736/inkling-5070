# Inkling 2 Mini: the update (update.bat runs a copy of this file from update_tmp\).
#   update.bat              check GitHub's newest release of Nsan736/inkling-5070, and update if it is newer
#   update.bat --check      only say whether a newer release is out
#   update.bat --rollback   put back the newest version kept in old\
# A download is used only if its SHA-256 is the one in the release's SHA256SUMS.txt.  Only the program is replaced:
# models\, config\, .venv\, logs\, data\, tmp\ and old\ (and anything else this file does not know) stay as they are.
param(
    [string]$Root,
    [string]$LocalDir = "",     # testing: a folder with the release's zip and SHA256SUMS.txt in place of GitHub
    [string]$LocalTag = "",     # testing: the version of that folder's release, e.g. v0.3.5
    [Parameter(ValueFromRemainingArguments = $true)] $Rest
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo = "Nsan736/inkling-5070"
$Api = "https://api.github.com/repos/$Repo/releases/latest"
$Downloads = "https://github.com/$Repo/releases/download/"
# What the program is: these names at the top of the folder (and whatever else the new package has there)
$Program = @("bin", "docs", "licenses", "scripts", "server", "setup.bat", "run.bat", "update.bat", "requirements.txt",
             "README.md", "README.en.md", "LICENSE", "NOTICE", "UPSTREAM.md", "VERSION")
# What is never touched
$Keep = @("models", "config", ".venv", "logs", "data", "tmp", "old", "update_tmp")

$Root = (Resolve-Path -LiteralPath $Root).Path
$Work = Join-Path $Root "update_tmp"
$check = $Rest -contains "--check"
$rollback = $Rest -contains "--rollback"

function Say($text) { Write-Host $text }
function Fail($text) {
    Write-Host "ERROR: $text"
    exit 1
}

function Current-Version {
    $f = Join-Path $Root "VERSION"
    if (Test-Path -LiteralPath $f) { return (Get-Content -LiteralPath $f -Raw).Trim() }
    # 0.3.4 and before had no VERSION file: the README names the version in its first lines
    $r = Join-Path $Root "README.md"
    if (Test-Path -LiteralPath $r) {
        $m = [regex]::Match((Get-Content -LiteralPath $r -Raw -Encoding UTF8), "\b0\.\d+\.\d+\b")
        if ($m.Success) { return $m.Value }
    }
    return "0.0.0"
}

function As-Version($text) {
    $t = $text.Trim().TrimStart("v", "V")
    try { return [version]$t } catch { Fail "not a version: $text" }
}

# The server or the engine of this folder running: its files are in use, and the update would break it
function Stop-If-Running {
    $mine = @()
    foreach ($p in Get-CimInstance Win32_Process) {
        $exe = [string]$p.ExecutablePath
        $cmd = [string]$p.CommandLine
        if (($exe -and $exe.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase) -and
             ($p.Name -like "strata-*")) -or
            ($cmd -and $cmd.IndexOf($Root, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
             ($cmd -match "server\.py" -or $cmd -match "run\.py"))) {
            $mine += "$($p.Name) (PID $($p.ProcessId))"
        }
    }
    if ($mine.Count -gt 0) {
        Say "The server or the engine of this folder is running:"
        $mine | ForEach-Object { Say "  $_" }
        Fail "stop it first (close run.bat's window, or Ctrl+C there), then run update.bat again. Nothing was changed."
    }
}

# Top-level names of the program now in the folder (the known ones, and those the new package has)
function Program-Names($extra) {
    $names = @($Program + $extra | Sort-Object -Unique)
    return @($names | Where-Object { ($Keep -notcontains $_) -and (Test-Path -LiteralPath (Join-Path $Root $_)) })
}

function Move-Program($names, $to) {
    New-Item -ItemType Directory -Force -Path $to | Out-Null
    $moved = @()
    try {
        foreach ($n in $names) {
            Move-Item -LiteralPath (Join-Path $Root $n) -Destination (Join-Path $to $n)
            $moved += $n
        }
    } catch {
        foreach ($n in $moved) { Move-Item -LiteralPath (Join-Path $to $n) -Destination (Join-Path $Root $n) }
        Fail "could not move $n ($($_.Exception.Message)); everything was put back. Is a file of it open?"
    }
}

# The settings the new version knows of that the user's config\inkling.json lacks: added with their defaults,
# the user's values kept
function Merge-Config {
    $cfgPath = Join-Path $Root "config\inkling.json"
    $defPath = Join-Path $Root "scripts\config_defaults.json"
    if (-not (Test-Path -LiteralPath $cfgPath) -or -not (Test-Path -LiteralPath $defPath)) { return }
    $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $def = Get-Content -LiteralPath $defPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $added = @()
    foreach ($p in $def.PSObject.Properties) {
        $have = $cfg.PSObject.Properties[$p.Name]
        if (-not $have) {
            $cfg | Add-Member -NotePropertyName $p.Name -NotePropertyValue $p.Value
            $added += "$($p.Name) = $($p.Value | ConvertTo-Json -Compress)"
        } elseif ($p.Value -is [pscustomobject] -and $have.Value -is [pscustomobject]) {
            foreach ($q in $p.Value.PSObject.Properties) {
                if (-not $have.Value.PSObject.Properties[$q.Name]) {
                    $have.Value | Add-Member -NotePropertyName $q.Name -NotePropertyValue $q.Value
                    $added += "$($p.Name).$($q.Name) = $($q.Value | ConvertTo-Json -Compress)"
                }
            }
        }
    }
    if ($added.Count -eq 0) {
        Say "Settings: nothing new; config\inkling.json is as it was."
        return
    }
    Copy-Item -LiteralPath $cfgPath -Destination "$cfgPath.before-update" -Force
    $json = $cfg | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText($cfgPath, $json, (New-Object Text.UTF8Encoding $false))
    Say "Settings: added to config\inkling.json (your values kept; the old file is config\inkling.json.before-update):"
    $added | ForEach-Object { Say "  $_" }
}

function Update-Venv($oldReq) {
    $newReq = Join-Path $Root "requirements.txt"
    $py = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $py) -or -not (Test-Path -LiteralPath $newReq)) { return }
    $same = (Test-Path -LiteralPath $oldReq) -and
            ((Get-FileHash -LiteralPath $oldReq -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $newReq -Algorithm SHA256).Hash)
    if ($same) {
        Say "Python packages: the same as before."
        return
    }
    Say "Python packages: requirements.txt changed; updating .venv ..."
    & $py -m pip install --disable-pip-version-check -q -r $newReq
    if ($LASTEXITCODE -ne 0) { Fail "pip failed; run setup.bat --configure-only to try again (the program is already updated)" }
}

# ---- --rollback ----
if ($rollback) {
    Stop-If-Running
    $oldDir = Join-Path $Root "old"
    $kept = @(Get-ChildItem -LiteralPath $oldDir -Directory -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -notlike "*.replaced*" -and $_.Name -match "^\d+\.\d+\.\d+" } |
              Sort-Object { [version]([regex]::Match($_.Name, "^\d+\.\d+\.\d+").Value) }, LastWriteTime -Descending)
    if ($kept.Count -eq 0) { Fail "old\ holds no version to put back" }
    $back = $kept[0]
    $cur = Current-Version
    $names = Program-Names @(Get-ChildItem -LiteralPath $back.FullName | ForEach-Object { $_.Name })
    $aside = Join-Path $oldDir "$cur.replaced"
    if (Test-Path -LiteralPath $aside) { $aside = Join-Path $oldDir ("$cur.replaced-" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
    Say "Putting back $($back.Name) (the program of $cur goes to $aside) ..."
    Move-Program $names $aside
    foreach ($item in Get-ChildItem -LiteralPath $back.FullName) {
        Move-Item -LiteralPath $item.FullName -Destination (Join-Path $Root $item.Name)
    }
    Remove-Item -LiteralPath $back.FullName -Recurse -Force
    # a version from before update.bat keeps this one, so that it can be updated again
    if (-not (Test-Path -LiteralPath (Join-Path $Root "update.bat"))) {
        Copy-Item -LiteralPath (Join-Path $aside "update.bat") -Destination (Join-Path $Root "update.bat")
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root "scripts\update.ps1"))) {
        Copy-Item -LiteralPath (Join-Path $aside "scripts\update.ps1") -Destination (Join-Path $Root "scripts\update.ps1")
    }
    Say "Done: version $($back.Name) is back. Start it with run.bat."
    exit 0
}

# ---- the newest release ----
$cur = Current-Version
Say "This folder: version $cur"
if ($LocalDir) {
    if (-not $LocalTag) { Fail "-LocalDir needs -LocalTag" }
    $tag = $LocalTag
    $zipAsset = Get-ChildItem -LiteralPath $LocalDir -Filter "inkling-5070-*-windows-x64-cuda13.zip" | Select-Object -First 1
    $sumAsset = Join-Path $LocalDir "SHA256SUMS.txt"
    if (-not $zipAsset -or -not (Test-Path -LiteralPath $sumAsset)) { Fail "$LocalDir has no release zip and SHA256SUMS.txt" }
    Say "Release (a local folder, for testing): $tag"
} else {
    try {
        $rel = Invoke-RestMethod -Uri $Api -Headers @{ "User-Agent" = "inkling-5070-update"; "Accept" = "application/vnd.github+json" } -TimeoutSec 30
    } catch {
        Fail "could not ask GitHub for the newest release ($($_.Exception.Message))"
    }
    $tag = [string]$rel.tag_name
    Say "Newest release on GitHub: $tag"
}
if ((As-Version $tag) -le (As-Version $cur)) {
    Say "Already up to date."
    exit 0
}
if ($check) {
    Say "A newer release is out: $tag. Run update.bat to update."
    exit 0
}
Stop-If-Running

if (Test-Path -LiteralPath (Join-Path $Work "pkg")) { Remove-Item -LiteralPath (Join-Path $Work "pkg") -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Work "pkg") | Out-Null
if ($LocalDir) {
    $zipName = $zipAsset.Name
    $zip = Join-Path $Work $zipName
    Copy-Item -LiteralPath $zipAsset.FullName -Destination $zip -Force
    $sums = Join-Path $Work "SHA256SUMS.txt"
    Copy-Item -LiteralPath $sumAsset -Destination $sums -Force
} else {
    $za = @($rel.assets | Where-Object { $_.name -like "inkling-5070-*-windows-x64-cuda13.zip" })
    $sa = @($rel.assets | Where-Object { $_.name -eq "SHA256SUMS.txt" })
    if ($za.Count -ne 1) { Fail "the release $tag has no single package zip. Nothing was changed." }
    if ($sa.Count -ne 1) { Fail "the release $tag has no SHA256SUMS.txt, so its zip cannot be checked. Nothing was changed." }
    foreach ($u in @($za[0].browser_download_url, $sa[0].browser_download_url)) {
        if (-not ([string]$u).StartsWith($Downloads)) { Fail "an asset is not on $Downloads ($u). Nothing was changed." }
    }
    $zipName = $za[0].name
    $zip = Join-Path $Work $zipName
    $sums = Join-Path $Work "SHA256SUMS.txt"
    Say "Downloading $zipName ($([math]::Round($za[0].size / 1MB)) MiB) ..."
    try {
        Invoke-WebRequest -Uri $sa[0].browser_download_url -OutFile $sums -UseBasicParsing -Headers @{ "User-Agent" = "inkling-5070-update" }
        Invoke-WebRequest -Uri $za[0].browser_download_url -OutFile $zip -UseBasicParsing -Headers @{ "User-Agent" = "inkling-5070-update" }
    } catch {
        Fail "the download failed ($($_.Exception.Message)). Nothing was changed."
    }
}

# ---- the SHA-256 ----
$want = $null
foreach ($line in Get-Content -LiteralPath $sums) {
    $f = $line.Trim() -split "\s+", 2
    if ($f.Count -eq 2 -and $f[1].TrimStart("*") -eq $zipName) { $want = $f[0].ToLower() }
}
if (-not $want) { Fail "SHA256SUMS.txt has no line for $zipName. Nothing was changed." }
$got = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
if ($got -ne $want) {
    Remove-Item -LiteralPath $zip -Force
    Fail "the SHA-256 of $zipName is $got, not $want as SHA256SUMS.txt says. The download was deleted; nothing was changed."
}
Say "SHA-256 checked: $got"

# ---- unpack, set the old program aside, put the new one in ----
Say "Unpacking ..."
Add-Type -AssemblyName System.IO.Compression.FileSystem
$pkg = Join-Path $Work "pkg"
[IO.Compression.ZipFile]::ExtractToDirectory($zip, $pkg)
$tops = @(Get-ChildItem -LiteralPath $pkg)
if ($tops.Count -ne 1 -or -not $tops[0].PSIsContainer) { Fail "the zip does not hold one folder. Nothing was changed." }
$new = $tops[0].FullName
$newNames = @(Get-ChildItem -LiteralPath $new | ForEach-Object { $_.Name } | Where-Object { $Keep -notcontains $_ })
if (-not (Test-Path -LiteralPath (Join-Path $new "bin\strata-inkling2.exe"))) { Fail "the zip has no bin\strata-inkling2.exe. Nothing was changed." }

$aside = Join-Path $Root "old\$cur"
if (Test-Path -LiteralPath $aside) { $aside = Join-Path $Root ("old\$cur-" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
$names = Program-Names $newNames
Say "Setting the program of $cur aside in $aside ..."
Move-Program $names $aside
try {
    foreach ($n in $newNames) {
        Move-Item -LiteralPath (Join-Path $new $n) -Destination (Join-Path $Root $n)
    }
} catch {
    Say "ERROR: could not put the new program in ($($_.Exception.Message)); putting $cur back ..."
    foreach ($n in $newNames) {
        $p = Join-Path $Root $n
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Recurse -Force }
    }
    foreach ($n in $names) { Move-Item -LiteralPath (Join-Path $aside $n) -Destination (Join-Path $Root $n) }
    exit 1
}
Set-Content -LiteralPath (Join-Path $Root "VERSION") -Value ((As-Version $tag).ToString()) -Encoding ASCII

Merge-Config
Update-Venv (Join-Path $aside "requirements.txt")
Remove-Item -LiteralPath $pkg -Recurse -Force
Remove-Item -LiteralPath $zip -Force
Say ""
Say "Done: version $((As-Version $tag).ToString()). Start it with run.bat."
Say "The program of $cur is in $aside; update.bat --rollback puts it back."
exit 0
