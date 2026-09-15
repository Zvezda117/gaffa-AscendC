#include "gaffa/dedispersion_ascend.h"
#include "gaffa/filterbank.h"
#include "gaffa/filterbank_view.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace {

struct Args {
  std::filesystem::path path;
  std::size_t ndm = 32;
  double dm_low = 0.0;
  double dm_step = 1.0;
  int iterations = 5;
  int device_id = 0;
  std::size_t subband_channels = 32;
  std::size_t ndm_per_nominal = 32;
  std::size_t time_tile_samples = 81920;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <filterbank.fil> [ndm] [dm_low] [dm_step] [iterations]"
            << " [device_id] [subband_channels] [ndm_per_nominal]"
            << " [time_tile_samples]\n";
}

Args parse_args(int argc, char** argv) {
  if (argc < 2 || argc > 10) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }
  Args args;
  args.path = argv[1];
  if (argc > 2) args.ndm = static_cast<std::size_t>(std::stoull(argv[2]));
  if (argc > 3) args.dm_low = std::stod(argv[3]);
  if (argc > 4) args.dm_step = std::stod(argv[4]);
  if (argc > 5) args.iterations = std::stoi(argv[5]);
  if (argc > 6) args.device_id = std::stoi(argv[6]);
  if (argc > 7) args.subband_channels = static_cast<std::size_t>(std::stoull(argv[7]));
  if (argc > 8) args.ndm_per_nominal = static_cast<std::size_t>(std::stoull(argv[8]));
  if (argc > 9) args.time_tile_samples = static_cast<std::size_t>(std::stoull(argv[9]));

  if (args.ndm == 0 || !(args.dm_step > 0.0) || !std::isfinite(args.dm_step) ||
      !std::isfinite(args.dm_low) || args.iterations <= 0 || args.device_id < 0 ||
      args.subband_channels == 0 || args.ndm_per_nominal == 0 ||
      args.time_tile_samples == 0) {
    throw std::invalid_argument("invalid benchmark options");
  }
  return args;
}

gaffa::MultiDmDedispersionPlan make_plan(
    const gaffa::FilterbankHeader& header, const Args& args) {
  const auto ref = *std::max_element(header.frequency_table.begin(),
                                     header.frequency_table.end());
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = args.dm_low,
      .dm_step = args.dm_step,
      .ndm = args.ndm,
      .ref_frequency_mhz = ref,
      .tsamp = header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(header.nchans),
  };
}

template <typename T>
void run_typed(const gaffa::FilterbankData& filterbank, const Args& args) {
  const auto samples = gaffa::sample_view<T>(filterbank);
  const auto plan = make_plan(filterbank.header, args);
  const gaffa::SubbandDedispersionOptions subband{
      .subband_channels = args.subband_channels,
      .ndm_per_nominal = args.ndm_per_nominal,
  };
  const gaffa::AscendDedispersionOptions ascend{
      .device_id = args.device_id,
      .block_dim = 0,
      .time_tile_samples = args.time_tile_samples,
  };

  double total = 0.0;
  std::size_t output_elements = 0;
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    auto result = gaffa::dedisperse_subband_ascend(
        samples, filterbank.header.frequency_table, plan, subband, ascend);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    total += seconds;
    output_elements = result.data.size();
    std::cout << "iteration=" << iteration << " seconds=" << std::setprecision(8)
              << seconds << " output_elements=" << output_elements << '\n';
  }

  const double mean = total / static_cast<double>(args.iterations);
  const double input_bytes = static_cast<double>(samples.data.size() * sizeof(T));
  std::cout << "summary backend=ascend-subband"
            << " device_id=" << args.device_id
            << " ndm=" << args.ndm
            << " nchans=" << filterbank.header.nchans
            << " nsamples=" << filterbank.header.nsamples
            << " mean_seconds=" << std::setprecision(8) << mean
            << " input_gib_per_second="
            << (input_bytes / mean / (1024.0 * 1024.0 * 1024.0))
            << " output_elements=" << output_elements << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const gaffa::FilterbankData filterbank = gaffa::read_filterbank(args.path);
    std::visit(
        [&](const auto& values) {
          using T = typename std::decay_t<decltype(values)>::value_type;
          run_typed<T>(filterbank, args);
        },
        filterbank.samples);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
