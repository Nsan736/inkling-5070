"""128K の参照ロジット (quality_all.py の手順 ref128k) だけを先に作る。作ったものは quality_all.py --run main が使う。"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quality_all as q  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
run = q.Run(sys.argv[1] if len(sys.argv) > 1 else "main", False)
if not run.done("ref128k"):
    run.log("手順 ref128k を先に作る (make_ref128k.py)")
    t0 = time.time()
    ref = q.make_ref(run, False)
    q.save_time("ref128k", time.time() - t0)
    run.put("ref128k", ref)
    run.log(f"手順 ref128k が終わった ({q.fmt_s(time.time() - t0)})")
