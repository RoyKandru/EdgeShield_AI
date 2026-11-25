#!/bin/bash
set -e
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++20 -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -fPIC -Wno-psabi -DET_BUILD_WITH_BLAS -pthread -march=armv8.2-a+dotprod \
  -DC10_USING_CUSTOM_GENERATED_MACROS -DCPUINFO_SUPPORTED_PLATFORM=1 -DET_USE_THREADPOOL -DEXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES -DKLEIDIAI_ERROR_TRAP=0 -DKLEIDIAI_HAS_BUILTIN_ASSUME0=0 -DKLEIDIAI_HAS_BUILTIN_TRAP=0 -DKLEIDIAI_HAS_BUILTIN_UNREACHABLE=1 -DPCRE2_STATIC -DPHASE1_WITH_EXECUTORCH=1 -DTK_LOG_LEVEL=Info -DXNN_LOG_LEVEL=0 \
  -I/home/open/executorch -I/home/open/executorch/.. -I/home/open/executorch/runtime/core/portable_type/c10 -I/home/open/edgeproc/bharat-ai/build-et/executorch/include -I/home/open/executorch/backends/xnnpack/third-party/cpuinfo/include -I/home/open/executorch/backends/xnnpack/third-party/pthreadpool/include -I/home/open/executorch/backends/xnnpack/third-party/XNNPACK/include -I/home/open/edgeproc/bharat-ai/build-et/schema/include -I/home/open/executorch/third-party/flatbuffers/include -I/home/open/edgeproc/bharat-ai/build-et/kleidiai-source -I/home/open/executorch/extension/llm/tokenizers/include -I/home/open/executorch/extension/llm/tokenizers/third-party/sentencepiece -I/home/open/executorch/extension/llm/tokenizers/third-party/sentencepiece/src -I/home/open/executorch/extension/llm/tokenizers/third-party/re2 -I/home/open/executorch/extension/llm/tokenizers/third-party/json/single_include -I/home/open/executorch/extension/llm/tokenizers/third-party/llama.cpp-unicode/include -I/home/open/executorch/extension/llm/tokenizers/third-party/abseil-cpp -I/home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/pcre2 \
  -c src/executorch_runner.cpp -o bin/executorch_runner.o && \
/usr/bin/c++ -O3 -DNDEBUG \
  -Wl,--whole-archive /home/open/edgeproc/bharat-ai/build-et/executorch/libexecutorch.a -Wl,--no-whole-archive \
  -Wl,--whole-archive /home/open/edgeproc/bharat-ai/build-et/executorch/configurations/liboptimized_native_cpu_ops_lib.a -Wl,--no-whole-archive \
  -Wl,--whole-archive /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/quantized/libquantized_ops_lib.a -Wl,--no-whole-archive \
  -Wl,--whole-archive /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/custom_ops/libcustom_ops.a -Wl,--no-whole-archive \
  -Wl,--whole-archive /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/libxnnpack_backend.a -Wl,--no-whole-archive \
  -Wl,--whole-archive /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/libregex_lookahead.a -Wl,--no-whole-archive \
  bin/executorch_runner.o -o bin/executorch_runner \
  /home/open/edgeproc/bharat-ai/build-et/executorch/libexecutorch.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/configurations/liboptimized_native_cpu_ops_lib.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/optimized/liboptimized_kernels.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/portable/libportable_kernels.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/optimized/libcpublas.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/optimized/libeigen_blas.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/quantized/libquantized_kernels.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/quantized/libquantized_ops_lib.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/custom_ops/libcustom_ops.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/libxnnpack_backend.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a \
  /home/open/edgeproc/bharat-ai/build-et/kleidiai/libkleidiai.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/threadpool/libextension_threadpool.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/runner/libextension_llm_runner.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/portable/liboptimized_portable_kernels.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/quantized/libquantized_kernels.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/optimized/libcpublas.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/optimized/libeigen_blas.a \
  /home/open/edgeproc/bharat-ai/build-et/kleidiai/libkleidiai.a -lm \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/threadpool/libextension_threadpool.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/kernels/portable/cpu/util/libkernels_util_all_deps.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/module/libextension_module.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/data_loader/libextension_data_loader.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/flat_tensor/libextension_flat_tensor.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/named_data_map/libextension_named_data_map.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/tensor/libextension_tensor.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/sampler/libextension_llm_sampler.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/memory_allocator/libextension_memory_allocator.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/libexecutorch_core.a -ldl \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/libtokenizers.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/sp-build/src/libsentencepiece.a -latomic \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/re2/libre2.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_check_op.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_leak_check.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_conditions.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_message.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_nullguard.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_examine_stack.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_format.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_proto.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_log_sink_set.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_globals.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_globals.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_sink.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_entry.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_strerror.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_vlog_config_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/log/libabsl_log_internal_fnmatch.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_marshalling.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_reflection.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_config.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_program_name.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_private_handle_accessor.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_commandlineflag.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/flags/libabsl_flags_commandlineflag_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/container/libabsl_raw_hash_set.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_cord.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_cordz_info.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_cord_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_cordz_functions.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_cordz_handle.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/crc/libabsl_crc_cord_state.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/crc/libabsl_crc32c.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/crc/libabsl_crc_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/crc/libabsl_crc_cpu_detect.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/container/libabsl_hashtablez_sampler.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/profiling/libabsl_exponential_biased.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/hash/libabsl_hash.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/hash/libabsl_city.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/types/libabsl_bad_variant_access.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/hash/libabsl_low_level_hash.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/types/libabsl_bad_optional_access.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_str_format_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/synchronization/libabsl_synchronization.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/synchronization/libabsl_graphcycles_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/synchronization/libabsl_kernel_timeout_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_stacktrace.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_symbolize.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_malloc_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_debugging_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_demangle_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_demangle_rust.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_decode_rust_punycode.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/debugging/libabsl_utf8_for_code_point.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_tracing_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/time/libabsl_time.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_strings.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_strings_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/strings/libabsl_string_view.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_base.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_spinlock_wait.a -lrt \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_throw_delegate.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_raw_logging_internal.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/base/libabsl_log_severity.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/numeric/libabsl_int128.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/time/libabsl_civil_time.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/abseil-cpp/absl/time/libabsl_time_zone.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/libregex_lookahead.a \
  /home/open/edgeproc/bharat-ai/build-et/executorch/extension/llm/tokenizers/third-party/pcre2/libpcre2-8.a

