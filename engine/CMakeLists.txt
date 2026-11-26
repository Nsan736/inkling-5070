# strata-inkling2: Inkling 2 Mini (Thinking Machines) engine for one NVIDIA card plus system RAM.
#
# The engine is PolyStrata's strata-poly (pull request #200, MIT) cut down to the Inkling 2 family; UPSTREAM.md lists
# every file and where it came from.  It is a project of its own, built beside Strata's engine and linked with none of
# its libraries: the kernels and headers it needs are this folder's copies (kernels/, base/, include/).  ggml is
# engine/third_party/strata-llama (llama.cpp 3cf03257, the pin PolyStrata uses): ggml-base and ggml-cpu for the CPU
# experts, and ggml-cuda's MMQ instances for the prompt.
#
#   cmake -S engine/src/inkling2 -B build-inkling2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
#   cmake --build build-inkling2 -j 12 --target strata-inkling2
#
# Every .cu is its own translation unit, so editing one kernel rebuilds that file and relinks.
cmake_minimum_required(VERSION 3.24)
project(strata_inkling2 VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_compile_definitions(STRATA_VERSION="inkling2-${PROJECT_VERSION}" STRATA_NATIVE_EXPERTS=1 STRATA_Q6K_EXPERTS=1)
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/Zi>
                      $<$<COMPILE_LANGUAGE:CXX>:/bigobj> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

set(INK_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INK_ENGINE_ROOT ${INK_DIR}/../..)
# inkling2 fork: zstd's decoder (BSD), for the manifests of .strata files (include/strata/sq/sq_reader.hpp)
set(INK_ZSTD_ROOT ${INK_ENGINE_ROOT}/third_party/zstd-dec CACHE PATH "zstd's decoder sources")
set(INK_INCLUDE ${INK_DIR}/include ${INK_ZSTD_ROOT})
set(INK_GGML_ROOT ${INK_ENGINE_ROOT}/third_party/strata-llama CACHE PATH "llama.cpp tree to take ggml from")
set(INK_GG ${INK_GGML_ROOT}/ggml)
set(INK_MMQ ${INK_GG}/src/ggml-cuda)

# ---- ggml (CPU only: the CUDA parts are compiled below, only the MMQ instances) ----
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# INK_NATIVE=ON tunes ggml-cpu to the building machine; OFF (for release packages) builds AVX2 + FMA + F16C, which every
# CPU the engine's AVX2 code (cpu_experts.cpp) needs has anyway
option(INK_NATIVE "ggml-cpu for this machine's CPU only" ON)
if(INK_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${f} ON CACHE BOOL "" FORCE)
  endforeach()
  foreach(f GGML_AVX512 GGML_AVX_VNNI GGML_AVX512_VNNI GGML_AVX512_VBMI GGML_AVX512_BF16)
    set(${f} OFF CACHE BOOL "" FORCE)
  endforeach()
endif()
add_subdirectory(${INK_GG} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

set(INK_CUDA_OPTS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
if(MSVC)
  list(APPEND INK_CUDA_OPTS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

add_library(ink_zstd STATIC
  ${INK_ZSTD_ROOT}/common/debug.c ${INK_ZSTD_ROOT}/common/entropy_common.c ${INK_ZSTD_ROOT}/common/error_private.c
  ${INK_ZSTD_ROOT}/common/fse_decompress.c ${INK_ZSTD_ROOT}/common/xxhash.c ${INK_ZSTD_ROOT}/common/zstd_common.c
  ${INK_ZSTD_ROOT}/decompress/huf_decompress.c ${INK_ZSTD_ROOT}/decompress/zstd_ddict.c
  ${INK_ZSTD_ROOT}/decompress/zstd_decompress.c ${INK_ZSTD_ROOT}/decompress/zstd_decompress_block.c)
target_include_directories(ink_zstd PUBLIC ${INK_ZSTD_ROOT})
target_compile_definitions(ink_zstd PRIVATE ZSTD_DISABLE_ASM=1)

# ---- ink_core: device, pinned memory, graphs, files, conversation files, drafting helpers ----
add_library(ink_core STATIC
  base/core/device.cu base/core/pinned.cu base/core/graph.cpp base/core/weights.cpp base/core/layout.cpp
  base/core/conversation_file.cpp base/core/conversation_memory.cpp
  base/platform/memory.cpp base/platform/direct_file.cpp
  base/spec/suffix_drafter.cpp base/spec/controller.cpp base/spec/draft_policy.cpp base/spec/draft_source.cpp
  base/kernels/ngram.cpp base/ngram/ple_reader.cpp)
target_include_directories(ink_core PUBLIC ${INK_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(ink_core PUBLIC CUDA::cudart ink_zstd)

# ---- ink_kernels: the quantized products and the small kernels (one .cu each) ----
set(INK_KERNELS
  native_mmvq iq_kernels dequant_bf16 elementwise quantize_act sampler native_router native_moe native_bf16
  native_rope rope bf16_gemv cvec kv_q8 native_flash_attn verify_kernels)
list(TRANSFORM INK_KERNELS PREPEND kernels/)
list(TRANSFORM INK_KERNELS APPEND .cu)
add_library(ink_kernels STATIC ${INK_KERNELS})
target_include_directories(ink_kernels PUBLIC ${INK_INCLUDE} PRIVATE ${INK_DIR}/kernels ${INK_DIR}/third_party_ggml)
target_link_libraries(ink_kernels PUBLIC ink_core)
# PolyStrata の設定どおり: llama.cpp の CUDA と同じ演算にそろえる native_* だけ fast math
set(INK_FAST_MATH_KERNELS ${INK_KERNELS})
list(FILTER INK_FAST_MATH_KERNELS INCLUDE REGEX "kernels/native_")
set_source_files_properties(${INK_FAST_MATH_KERNELS} PROPERTIES COMPILE_OPTIONS "--use_fast_math")

# ---- ink_mmq: llama.cpp's MMQ tiles for the prompt (ggml-cuda, MIT) and PolyStrata's NVFP4 int8 tiles ----
set(INK_MMQ_INSTANCES "")
foreach(_t q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q2_0)
  list(APPEND INK_MMQ_INSTANCES ${INK_MMQ}/template-instances/mmq-instance-${_t}.cu)
endforeach()
# MXFP4 / NVFP4 instances need the arch-specific target (12x -> 12xa) for the FP4 MMA with block scales
set(INK_FP4_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
list(TRANSFORM INK_FP4_ARCHS REPLACE "^(12[0-9])$" "\\1a")
list(TRANSFORM INK_FP4_ARCHS REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
add_library(ink_mmq_fp4 OBJECT ${INK_MMQ}/template-instances/mmq-instance-mxfp4.cu
                               ${INK_MMQ}/template-instances/mmq-instance-nvfp4.cu)
set_target_properties(ink_mmq_fp4 PROPERTIES CUDA_ARCHITECTURES "${INK_FP4_ARCHS}")
target_include_directories(ink_mmq_fp4 PRIVATE ${INK_GG}/include ${INK_GG}/src ${INK_MMQ})
target_compile_options(ink_mmq_fp4 PRIVATE ${INK_CUDA_OPTS})
# inkling2 fork, INKLING2_SITU_MMQ: SiTU into the MMQ Q8_1 rows (situ_mmq.cu).  -use_fast_math as ink_mmq's but for
# its function mapping, so that the quantization is quantize.cu's to the bit and SiTU's tanhf/expf the engine's
add_library(ink_situ_mmq OBJECT situ_mmq.cu)
target_include_directories(ink_situ_mmq PRIVATE ${INK_GG}/include ${INK_GG}/src ${INK_MMQ} ${INK_INCLUDE})
target_compile_options(ink_situ_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -ftz=true -prec-div=false
                       -prec-sqrt=false>)
add_library(ink_mmq STATIC mmq.cu fattn.cu mmq_nvfp4.cu base/prefill/ggml_cuda_host.cu ${INK_MMQ}/quantize.cu
                           ${INK_MMQ_INSTANCES} $<TARGET_OBJECTS:ink_mmq_fp4> $<TARGET_OBJECTS:ink_situ_mmq>)
target_include_directories(ink_mmq PRIVATE ${INK_GG}/include ${INK_GG}/src ${INK_MMQ} ${INK_INCLUDE})
target_compile_options(ink_mmq PRIVATE ${INK_CUDA_OPTS})
target_compile_definitions(ink_mmq PRIVATE STRATA_POLY_NVFP4=1)
target_link_libraries(ink_mmq PUBLIC ggml-base CUDA::cudart)

# inkling2 fork: Strata's sm_120a SF4 prompt kernels (W4A4, INKLING2_SQ_W4A4, opt-in), the same 12xa archs
add_library(ink_sq_fp4 STATIC strata_fp4/ink_moe_fp4_sm120.cu strata_fp4/sq_w4a4.cu)
set_target_properties(ink_sq_fp4 PROPERTIES CUDA_ARCHITECTURES "${INK_FP4_ARCHS}")
target_include_directories(ink_sq_fp4 PRIVATE strata_fp4 ${INK_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(ink_sq_fp4 PRIVATE ${INK_CUDA_OPTS})

# ---- ink_cpu: the routed experts on the CPU (ggml-cpu dot products) ----
add_library(ink_cpu STATIC cpu_experts.cpp cpu_lowbit.cpp sq/cpu_sq.cpp)
target_include_directories(ink_cpu PUBLIC ${INK_GG}/include ${INK_INCLUDE})
set_source_files_properties(cpu_lowbit.cpp PROPERTIES INCLUDE_DIRECTORIES ${INK_GG}/src)
target_link_libraries(ink_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)
if(MSVC)   # cpu_nvfp4.hpp's AVX2 / AVX-VNNI / F16C (the experts run on this PC's CPU)
  set_source_files_properties(cpu_experts.cpp sq/cpu_sq.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
endif()

# ---- the engine ----
set(INK_ENGINE_SOURCES engine.cpp inkling.cpp spill.cpp kernels.cu kernels_inkling.cu dspark.cu kernels_sq.cu)
set(INK_LIBS ink_sq_fp4 ink_mmq ink_cpu ink_kernels ink_core CUDA::cublas CUDA::cudart CUDA::cuda_driver)

# the engine's own sources are compiled once and shared by both programs
add_library(ink_engine OBJECT ${INK_ENGINE_SOURCES})
target_include_directories(ink_engine PUBLIC ${INK_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS} ${INK_GG}/include)
target_link_libraries(ink_engine PUBLIC ${INK_LIBS})

add_executable(strata-inkling2 main.cpp)
target_link_libraries(strata-inkling2 PRIVATE ink_engine)

# the logits of a token list, to compare with llama.cpp's (bench/compare_logits.py)
add_executable(inkling2_logits_dump logits_dump.cpp)
target_link_libraries(inkling2_logits_dump PRIVATE ink_engine)

# the Inkling kernels against a float reference, and with `bench` / `decode` their timings at decode shapes, among them
# a token's full-layer attention over a long q8_0 cache against the card's memory bandwidth (PolyStrata's test)
add_executable(inkling2_kernels_test inkling_kernels_test.cu kernels_inkling.cu kernels.cu)
target_include_directories(inkling2_kernels_test PRIVATE ${INK_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(inkling2_kernels_test PRIVATE ink_kernels CUDA::cublas CUDA::cudart)

# what the CPU side can reach apart from the engine: memory reads and ggml-cpu's NVFP4 dot products, by thread count
add_executable(inkling2_cpu_bandwidth cpu_bandwidth.cpp)
target_link_libraries(inkling2_cpu_bandwidth PRIVATE ggml-cpu ggml-base Threads::Threads)
if(MSVC)
  target_compile_options(inkling2_cpu_bandwidth PRIVATE /arch:AVX2)
endif()

# the engine's CPU experts called as a written token calls them, without the card (cpu_calls.cpp)
add_executable(inkling2_cpu_calls cpu_calls.cpp)
target_link_libraries(inkling2_cpu_calls PRIVATE ink_cpu)
if(MSVC)
  target_compile_options(inkling2_cpu_calls PRIVATE /arch:AVX2)
endif()

# host -> card copy speed from pinned and pageable memory by copy size (the prompt path's expert staging)
add_executable(inkling2_pcie_bandwidth pcie_bandwidth.cu)
target_link_libraries(inkling2_pcie_bandwidth PRIVATE CUDA::cudart)

# inkling2 fork (development): a prompt batch's full-layer attention over a long cache, timed apart from the engine
add_executable(inkling2_attn_bench inkling_attn_bench.cu kernels_inkling.cu kernels.cu)
target_include_directories(inkling2_attn_bench PRIVATE ${INK_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(inkling2_attn_bench PRIVATE ink_kernels CUDA::cublas CUDA::cudart)
