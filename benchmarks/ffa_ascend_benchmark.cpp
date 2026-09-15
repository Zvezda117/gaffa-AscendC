#include "gaffa/ascend_memory.h"
#include "gaffa/ascend_runtime.h"
#include "gaffa/ffa_ascend.h"
#include "gaffa/ffa_plan.h"

#include <acl/acl.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::size_t nseries = 8;
  std::size_t nsamples = 262144;
  int iterations = 5;
  int device_id = 0;
  double tsamp = 0.001;
  double period_min = 0.2;
  double period_max = 2.0;
  float snr_threshold = 7.0F;
};

Args parse_args(int argc, char** argv) {
  if (argc > 8) {
    throw std::invalid_argument(
        "Usage: ffa_ascend_benchmark [nseries] [nsamples] [iterations] "
        "[device_id] [tsamp] [period_min] [period_max]");
  }
  Args args;
  if (argc > 1) args.nseries = static_cast<std::size_t>(std::stoull(argv[1]));
  if (argc > 2) args.nsamples = static_cast<std::size_t>(std::stoull(argv[2]));
  if (argc > 3) args.iterations = std::stoi(argv[3]);
  if (argc > 4) args.device_id = std::stoi(argv[4]);
  if (argc > 5) args.tsamp = std::stod(argv[5]);
  if (argc > 6) args.period_min = std::stod(argv[6]);
  if (argc > 7) args.period_max = std::stod(argv[7]);
  if (args.nseries == 0 || args.nsamples == 0 || args.iterations <= 0 ||
      args.device_id < 0 || !(args.tsamp > 0.0) ||
      !(args.period_min > args.tsamp * 128.0) ||
      !(args.period_max > args.period_min)) {
    throw std::invalid_argument("invalid FFA benchmark options");
  }
  return args;
}

void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::vector<float> make_input(std::size_t count) {
  std::mt19937 generator(0x47414646U);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> values(count);
  for (float& value : values) {
    value = normal(generator);
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    gaffa::AscendRuntime runtime(args.device_id);

    const std::size_t input_count = args.nseries * args.nsamples;
    const std::vector<float> host = make_input(input_count);
    gaffa::AscendDeviceBuffer<float> device(input_count, args.device_id);
    check_acl(aclrtMemcpy(device.data(), device.bytes(), host.data(),
                          host.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy FFA benchmark input");

    const gaffa::FfaSearchPlan plan = gaffa::make_riptide_ffa_plan(
        args.nsamples, args.tsamp,
        gaffa::RiptideFfaPlanOptions{
            .period_min = args.period_min,
            .period_max = args.period_max,
            .bins_min = 128,
            .bins_max = 256,
            .min_periods = 1,
            .duty_cycle_max = 0.20,
            .width_trial_spacing = 1.5,
            .max_tasks = 1'000'000,
        });

    gaffa::AscendFfaProgram program(
        plan,
        gaffa::AscendFfaProgramOptions{.device_id = args.device_id},
        gaffa::AscendFfaExecutionOptions{
            .series_tile_size = args.nseries,
            .workspace_bytes_limit = 0,
            .block_dim = 0,
            .stream = runtime.stream(),
        });

    const gaffa::AscendTimeSeriesBatchView batch{
        .data = device.data(),
        .nseries = args.nseries,
        .nsamples = args.nsamples,
        .device_id = args.device_id,
    };
    const gaffa::FfaSearchOptions search{
        .snr_threshold = args.snr_threshold,
        .max_peaks = 0,
    };

    // Warm one execution so the reported loop focuses on steady-state reuse of
    // the Program workspace rather than first-use initialization.
    auto warmup = gaffa::run_ffa_batch_ascend(program, batch, search);
    runtime.synchronize();
    std::size_t peak_count = warmup.peaks.size();

    double total = 0.0;
    for (int iteration = 0; iteration < args.iterations; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      auto result = gaffa::run_ffa_batch_ascend(program, batch, search);
      runtime.synchronize();
      const auto end = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(end - begin).count();
      total += seconds;
      peak_count = result.peaks.size();
      std::cout << "iteration=" << iteration << " seconds=" << std::setprecision(8)
                << seconds << " peaks=" << peak_count << '\n';
    }

    const double mean = total / static_cast<double>(args.iterations);
    const double samples_per_second =
        static_cast<double>(args.nseries * args.nsamples) / mean;
    std::cout << "summary backend=ascend"
              << " device_id=" << args.device_id
              << " nseries=" << args.nseries
              << " nsamples=" << args.nsamples
              << " tasks=" << plan.tasks.size()
              << " mean_seconds=" << std::setprecision(8) << mean
              << " samples_per_second=" << samples_per_second
              << " peaks=" << peak_count
              << " workspace_bytes=" << program.workspace_shape().total_bytes
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
