#include "dedispersion_bindings.h"

#include "filterbank_bindings.h"
#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_ascend.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/sample_view.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

enum class PythonDedispersionBackend { Cpu, Ascend };

struct PyDedispersedResult {
  py::array data;
  std::string backend;
  double dm_low = 0.0;
  double dm_step = 0.0;
  std::size_t ndm = 0;
  std::size_t nsamples = 0;
  double tsamp = 0.0;
};

struct PyDedispersedSpectrum {
  py::array data;
  std::string backend;
  double dm = 0.0;
  std::size_t nsamples = 0;
  std::size_t nchans = 0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
};

PythonDedispersionBackend parse_backend(const std::string& value) {
  if (value == "cpu") {
    return PythonDedispersionBackend::Cpu;
  }
  if (value == "ascend") {
    return PythonDedispersionBackend::Ascend;
  }
  throw py::value_error("unknown dedispersion backend: " + value);
}

std::string backend_name(PythonDedispersionBackend backend) {
  return backend == PythonDedispersionBackend::Cpu ? "cpu" : "ascend";
}

py::array require_numpy_array(const py::object& value, const char* name) {
  if (!py::isinstance<py::array>(value)) {
    throw py::type_error(std::string(name) + " must be a numpy.ndarray");
  }
  return py::reinterpret_borrow<py::array>(value);
}

template <typename T>
bool buffer_matches_dtype(const py::buffer_info& info) {
  return info.itemsize == static_cast<py::ssize_t>(sizeof(T)) &&
         info.format == py::format_descriptor<T>::format();
}

void validate_2d_c_contiguous(const py::buffer_info& info, const char* name) {
  if (info.ndim != 2) {
    throw py::value_error(std::string(name) + " must be a 2D array");
  }
  if (info.shape[0] <= 0 || info.shape[1] <= 0) {
    throw py::value_error(std::string(name) + " must not be empty");
  }
  if (info.strides[1] != static_cast<py::ssize_t>(info.itemsize) ||
      info.strides[0] != info.shape[1] * info.strides[1]) {
    throw py::value_error(std::string(name) + " must be C-contiguous");
  }
}

void validate_positive_tsamp(double tsamp) {
  if (!std::isfinite(tsamp) || !(tsamp > 0.0)) {
    throw py::value_error("tsamp must be finite and positive");
  }
}

PyDedispersedResult make_external_dedispersed_result(
    const py::object& data_object, double tsamp, double dm_low,
    double dm_step, std::string backend) {
  validate_positive_tsamp(tsamp);
  if (!std::isfinite(dm_low) || dm_low < 0.0) {
    throw py::value_error("dm_low must be finite and non-negative");
  }
  if (!std::isfinite(dm_step) || dm_step < 0.0) {
    throw py::value_error("dm_step must be finite and non-negative");
  }
  py::array data = require_numpy_array(data_object, "data");
  const py::buffer_info info = data.request();
  validate_2d_c_contiguous(info, "data");
  if (!buffer_matches_dtype<std::uint32_t>(info) &&
      !buffer_matches_dtype<float>(info)) {
    throw py::type_error("DedispersedResult data must have dtype uint32 or float32");
  }
  return PyDedispersedResult{
      .data = std::move(data),
      .backend = std::move(backend),
      .dm_low = dm_low,
      .dm_step = dm_step,
      .ndm = static_cast<std::size_t>(info.shape[0]),
      .nsamples = static_cast<std::size_t>(info.shape[1]),
      .tsamp = tsamp,
  };
}

PyDedispersedSpectrum make_external_dedispersed_spectrum(
    const py::object& data_object, double tsamp, double dm,
    py::ssize_t chan_begin, std::optional<py::ssize_t> chan_end,
    std::string backend) {
  validate_positive_tsamp(tsamp);
  if (!std::isfinite(dm) || dm < 0.0) {
    throw py::value_error("dm must be finite and non-negative");
  }
  if (chan_begin < 0 || (chan_end.has_value() && *chan_end < 0)) {
    throw py::value_error("channel bounds must be non-negative");
  }
  py::array data = require_numpy_array(data_object, "data");
  const py::buffer_info info = data.request();
  validate_2d_c_contiguous(info, "data");
  if (!buffer_matches_dtype<std::uint8_t>(info) &&
      !buffer_matches_dtype<std::uint16_t>(info) &&
      !buffer_matches_dtype<float>(info)) {
    throw py::type_error(
        "DedispersedSpectrum data must have dtype uint8, uint16, or float32");
  }
  const auto nchans = static_cast<std::size_t>(info.shape[1]);
  const auto begin = static_cast<std::size_t>(chan_begin);
  if (!chan_end.has_value() &&
      nchans > std::numeric_limits<std::size_t>::max() - begin) {
    throw std::overflow_error("chan_end exceeds size_t range");
  }
  const std::size_t end = chan_end.has_value()
                              ? static_cast<std::size_t>(*chan_end)
                              : begin + nchans;
  if (end < begin || end - begin != nchans) {
    throw py::value_error("chan_end - chan_begin must equal data.shape[1]");
  }
  return PyDedispersedSpectrum{
      .data = std::move(data),
      .backend = std::move(backend),
      .dm = dm,
      .nsamples = static_cast<std::size_t>(info.shape[0]),
      .nchans = nchans,
      .tsamp = tsamp,
      .chan_begin = begin,
      .chan_end = end,
  };
}

void validate_filterbank_header(const gaffa::FilterbankHeader& header) {
  if (header.nsamples <= 0 || header.nchans <= 0) {
    throw std::invalid_argument("dedispersion requires non-empty filterbank data");
  }
  if (header.nifs != 1) {
    throw std::invalid_argument("dedispersion requires nifs == 1");
  }
  if (!(header.tsamp > 0.0) || !std::isfinite(header.tsamp)) {
    throw std::invalid_argument("filterbank tsamp must be finite and positive");
  }
  if (header.frequency_table.size() != static_cast<std::size_t>(header.nchans)) {
    throw std::invalid_argument("frequency table length must match nchans");
  }
}

double reference_frequency_mhz(const gaffa::FilterbankHeader& header) {
  validate_filterbank_header(header);
  return *std::max_element(header.frequency_table.begin(),
                           header.frequency_table.end());
}

template <typename T>
gaffa::HostSampleView<T> typed_sample_view(
    const gaffa::python::PyFilterbank& filterbank) {
  const gaffa::SampleShape shape = gaffa::sample_shape(filterbank.header);
  const py::buffer_info info = filterbank.data.request();
  if (info.ndim != 3 ||
      info.shape[0] != static_cast<py::ssize_t>(shape.nsamples) ||
      info.shape[1] != static_cast<py::ssize_t>(shape.nifs) ||
      info.shape[2] != static_cast<py::ssize_t>(shape.nchans)) {
    throw std::invalid_argument("filterbank data shape does not match header");
  }
  if (!buffer_matches_dtype<T>(info)) {
    throw std::invalid_argument("filterbank data dtype does not match header");
  }
  const py::ssize_t item = static_cast<py::ssize_t>(sizeof(T));
  if (info.strides[2] != item ||
      info.strides[1] != static_cast<py::ssize_t>(shape.nchans) * item ||
      info.strides[0] != static_cast<py::ssize_t>(shape.nifs * shape.nchans) * item) {
    throw std::invalid_argument(
        "filterbank data must be C-contiguous in (time, if, channel) order");
  }
  return gaffa::make_host_sample_view<T>(
      std::span<const T>(static_cast<const T*>(info.ptr),
                         gaffa::sample_element_count(shape)),
      shape);
}

template <typename T>
py::array result_array(gaffa::DedispersedResult<T>&& result) {
  auto owner = std::make_unique<std::vector<T>>(std::move(result.data));
  T* ptr = owner->data();
  py::capsule capsule(owner.release(), [](void* value) {
    delete static_cast<std::vector<T>*>(value);
  });
  const auto ndm = static_cast<py::ssize_t>(result.shape.ndm);
  const auto nsamples = static_cast<py::ssize_t>(result.shape.nsamples);
  const auto item = static_cast<py::ssize_t>(sizeof(T));
  return py::array_t<T>({ndm, nsamples}, {nsamples * item, item}, ptr, capsule);
}

template <typename T>
py::array spectrum_array(gaffa::DedispersedSpectrum<T>&& result) {
  auto owner = std::make_unique<std::vector<T>>(std::move(result.data));
  T* ptr = owner->data();
  py::capsule capsule(owner.release(), [](void* value) {
    delete static_cast<std::vector<T>*>(value);
  });
  const auto nsamples = static_cast<py::ssize_t>(result.shape.nsamples);
  const auto nchans = static_cast<py::ssize_t>(result.shape.nchans);
  const auto item = static_cast<py::ssize_t>(sizeof(T));
  return py::array_t<T>({nsamples, nchans}, {nchans * item, item}, ptr, capsule);
}

template <typename T>
PyDedispersedResult wrap_result(gaffa::DedispersedResult<T>&& result,
                                PythonDedispersionBackend backend,
                                double dm_low, double dm_step, double tsamp) {
  const auto ndm = result.shape.ndm;
  const auto nsamples = result.shape.nsamples;
  return PyDedispersedResult{
      .data = result_array(std::move(result)),
      .backend = backend_name(backend),
      .dm_low = dm_low,
      .dm_step = dm_step,
      .ndm = ndm,
      .nsamples = nsamples,
      .tsamp = tsamp,
  };
}

template <typename T>
PyDedispersedSpectrum wrap_spectrum(
    gaffa::DedispersedSpectrum<T>&& result,
    PythonDedispersionBackend backend, double tsamp) {
  const auto nsamples = result.shape.nsamples;
  const auto nchans = result.shape.nchans;
  const auto dm = result.dm;
  const auto chan_begin = result.chan_begin;
  const auto chan_end = result.chan_end;
  return PyDedispersedSpectrum{
      .data = spectrum_array(std::move(result)),
      .backend = backend_name(backend),
      .dm = dm,
      .nsamples = nsamples,
      .nchans = nchans,
      .tsamp = tsamp,
      .chan_begin = chan_begin,
      .chan_end = chan_end,
  };
}

gaffa::SingleDmDedispersionPlan single_plan(
    const gaffa::FilterbankHeader& header, double dm) {
  return {.dm = dm,
          .ref_frequency_mhz = reference_frequency_mhz(header),
          .tsamp = header.tsamp,
          .chan_begin = 0,
          .chan_end = static_cast<std::size_t>(header.nchans)};
}

gaffa::MultiDmDedispersionPlan multi_plan(
    const gaffa::FilterbankHeader& header, double dm_low,
    double dm_step, std::size_t ndm) {
  return {.dm_low = dm_low,
          .dm_step = dm_step,
          .ndm = ndm,
          .ref_frequency_mhz = reference_frequency_mhz(header),
          .tsamp = header.tsamp,
          .chan_begin = 0,
          .chan_end = static_cast<std::size_t>(header.nchans)};
}

gaffa::AscendDedispersionOptions ascend_options(
    int device_id, std::uint32_t block_dim, std::size_t time_tile_samples) {
  if (device_id < 0) {
    throw py::value_error("device_id must be non-negative");
  }
  if (time_tile_samples == 0) {
    throw py::value_error("time_tile_samples must be positive");
  }
  return {.device_id = device_id,
          .block_dim = block_dim,
          .time_tile_samples = time_tile_samples};
}

template <typename T>
PyDedispersedSpectrum spectrum_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    PythonDedispersionBackend backend,
    const gaffa::AscendDedispersionOptions& options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = single_plan(filterbank.header, dm);
  if (backend == PythonDedispersionBackend::Cpu) {
    gaffa::DedispersedSpectrum<T> result;
    { py::gil_scoped_release release;
      result = gaffa::dedisperse_spectrum_cpu(
          samples, filterbank.header.frequency_table, plan); }
    return wrap_spectrum(std::move(result), backend, filterbank.header.tsamp);
  }
  gaffa::DedispersedSpectrum<T> result;
  { py::gil_scoped_release release;
    result = gaffa::dedisperse_spectrum_ascend(
        samples, filterbank.header.frequency_table, plan, options); }
  return wrap_spectrum(std::move(result), backend, filterbank.header.tsamp);
}

template <typename T>
PyDedispersedResult single_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    PythonDedispersionBackend backend,
    const gaffa::AscendDedispersionOptions& options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = single_plan(filterbank.header, dm);
  if (backend == PythonDedispersionBackend::Cpu) {
    auto result = [&] {
      py::gil_scoped_release release;
      return gaffa::dedisperse_single_dm_cpu(
          samples, filterbank.header.frequency_table, plan);
    }();
    return wrap_result(std::move(result), backend, dm, 0.0,
                       filterbank.header.tsamp);
  }
  auto result = [&] {
    py::gil_scoped_release release;
    return gaffa::dedisperse_single_dm_ascend(
        samples, filterbank.header.frequency_table, plan, options);
  }();
  return wrap_result(std::move(result), backend, dm, 0.0,
                     filterbank.header.tsamp);
}

template <typename T>
PyDedispersedResult multi_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, PythonDedispersionBackend backend,
    const gaffa::AscendDedispersionOptions& options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = multi_plan(filterbank.header, dm_low, dm_step, ndm);
  if (backend == PythonDedispersionBackend::Cpu) {
    auto result = [&] {
      py::gil_scoped_release release;
      return gaffa::dedisperse_multi_dm_cpu(
          samples, filterbank.header.frequency_table, plan);
    }();
    return wrap_result(std::move(result), backend, dm_low, dm_step,
                       filterbank.header.tsamp);
  }
  auto result = [&] {
    py::gil_scoped_release release;
    return gaffa::dedisperse_multi_dm_ascend(
        samples, filterbank.header.frequency_table, plan, options);
  }();
  return wrap_result(std::move(result), backend, dm_low, dm_step,
                     filterbank.header.tsamp);
}

template <typename T>
PyDedispersedResult subband_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, PythonDedispersionBackend backend,
    const gaffa::SubbandDedispersionOptions& subband_options,
    const gaffa::AscendDedispersionOptions& options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = multi_plan(filterbank.header, dm_low, dm_step, ndm);
  if (backend == PythonDedispersionBackend::Cpu) {
    auto result = [&] {
      py::gil_scoped_release release;
      return gaffa::dedisperse_subband_cpu(
          samples, filterbank.header.frequency_table, plan, subband_options);
    }();
    return wrap_result(std::move(result), backend, dm_low, dm_step,
                       filterbank.header.tsamp);
  }
  auto result = [&] {
    py::gil_scoped_release release;
    return gaffa::dedisperse_subband_ascend(
        samples, filterbank.header.frequency_table, plan, subband_options,
        options);
  }();
  return wrap_result(std::move(result), backend, dm_low, dm_step,
                     filterbank.header.tsamp);
}

template <typename Func>
auto dispatch_dtype(const gaffa::python::PyFilterbank& filterbank, Func&& func) {
  validate_filterbank_header(filterbank.header);
  switch (filterbank.header.nbits) {
    case 8: return func.template operator()<std::uint8_t>();
    case 16: return func.template operator()<std::uint16_t>();
    case 32: return func.template operator()<float>();
    default: throw py::type_error("unsupported filterbank sample dtype");
  }
}

PyDedispersedSpectrum dedisperse_spectrum_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    const std::string& backend, int device_id, std::uint32_t block_dim,
    std::size_t time_tile_samples) {
  const auto selected = parse_backend(backend);
  const auto options = ascend_options(device_id, block_dim, time_tile_samples);
  return dispatch_dtype(filterbank, [&]<typename T>() {
    return spectrum_typed<T>(filterbank, dm, selected, options);
  });
}

PyDedispersedResult dedisperse_single_dm_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    const std::string& backend, int device_id, std::uint32_t block_dim,
    std::size_t time_tile_samples) {
  const auto selected = parse_backend(backend);
  const auto options = ascend_options(device_id, block_dim, time_tile_samples);
  return dispatch_dtype(filterbank, [&]<typename T>() {
    return single_typed<T>(filterbank, dm, selected, options);
  });
}

PyDedispersedResult dedisperse_multi_dm_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, const std::string& backend,
    int device_id, std::uint32_t block_dim, std::size_t time_tile_samples) {
  const auto selected = parse_backend(backend);
  const auto options = ascend_options(device_id, block_dim, time_tile_samples);
  return dispatch_dtype(filterbank, [&]<typename T>() {
    return multi_typed<T>(filterbank, dm_low, dm_step, ndm, selected, options);
  });
}

PyDedispersedResult dedisperse_subband_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, const std::string& backend,
    std::size_t subband_channels, std::size_t ndm_per_nominal,
    int device_id, std::uint32_t block_dim, std::size_t time_tile_samples) {
  const auto selected = parse_backend(backend);
  const gaffa::SubbandDedispersionOptions subband_options{
      .subband_channels = subband_channels,
      .ndm_per_nominal = ndm_per_nominal,
  };
  const auto options = ascend_options(device_id, block_dim, time_tile_samples);
  return dispatch_dtype(filterbank, [&]<typename T>() {
    return subband_typed<T>(filterbank, dm_low, dm_step, ndm, selected,
                            subband_options, options);
  });
}

std::string result_repr(const PyDedispersedResult& result) {
  return "<DedispersedResult shape=(" + std::to_string(result.ndm) + ", " +
         std::to_string(result.nsamples) + ") backend=" + result.backend + ">";
}

std::string spectrum_repr(const PyDedispersedSpectrum& result) {
  return "<DedispersedSpectrum shape=(" + std::to_string(result.nsamples) +
         ", " + std::to_string(result.nchans) + ") backend=" + result.backend +
         ">";
}

}  // namespace

namespace gaffa::python {

void bind_dedispersion(py::module_& module) {
  py::class_<PyDedispersedResult>(module, "DedispersedResult")
      .def(py::init(&make_external_dedispersed_result), py::arg("data"),
           py::kw_only(), py::arg("tsamp"), py::arg("dm_low") = 0.0,
           py::arg("dm_step") = 0.0, py::arg("backend") = "external")
      .def_readonly("data", &PyDedispersedResult::data)
      .def_readonly("backend", &PyDedispersedResult::backend)
      .def_readonly("dm_low", &PyDedispersedResult::dm_low)
      .def_readonly("dm_step", &PyDedispersedResult::dm_step)
      .def_readonly("ndm", &PyDedispersedResult::ndm)
      .def_readonly("nsamples", &PyDedispersedResult::nsamples)
      .def_readonly("tsamp", &PyDedispersedResult::tsamp)
      .def_property_readonly("shape", [](const PyDedispersedResult& value) {
        return py::make_tuple(value.ndm, value.nsamples);
      })
      .def_property_readonly("dtype", [](const PyDedispersedResult& value) {
        return value.data.attr("dtype");
      })
      .def_property_readonly("nbytes", [](const PyDedispersedResult& value) {
        return value.data.attr("nbytes");
      })
      .def("__repr__", &result_repr);

  py::class_<PyDedispersedSpectrum>(module, "DedispersedSpectrum")
      .def(py::init(&make_external_dedispersed_spectrum), py::arg("data"),
           py::kw_only(), py::arg("tsamp"), py::arg("dm") = 0.0,
           py::arg("chan_begin") = 0, py::arg("chan_end") = py::none(),
           py::arg("backend") = "external")
      .def_readonly("data", &PyDedispersedSpectrum::data)
      .def_readonly("backend", &PyDedispersedSpectrum::backend)
      .def_readonly("dm", &PyDedispersedSpectrum::dm)
      .def_readonly("nsamples", &PyDedispersedSpectrum::nsamples)
      .def_readonly("nchans", &PyDedispersedSpectrum::nchans)
      .def_readonly("tsamp", &PyDedispersedSpectrum::tsamp)
      .def_readonly("chan_begin", &PyDedispersedSpectrum::chan_begin)
      .def_readonly("chan_end", &PyDedispersedSpectrum::chan_end)
      .def_property_readonly("shape", [](const PyDedispersedSpectrum& value) {
        return py::make_tuple(value.nsamples, value.nchans);
      })
      .def_property_readonly("dtype", [](const PyDedispersedSpectrum& value) {
        return value.data.attr("dtype");
      })
      .def_property_readonly("nbytes", [](const PyDedispersedSpectrum& value) {
        return value.data.attr("nbytes");
      })
      .def("__repr__", &spectrum_repr);

  module.def("dedisperse_spectrum", &dedisperse_spectrum_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm"),
             py::arg("backend") = "cpu", py::arg("device_id") = 0,
             py::arg("block_dim") = 0, py::arg("time_tile_samples") = 81920);
  module.def("dedisperse_single_dm", &dedisperse_single_dm_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm"),
             py::arg("backend") = "cpu", py::arg("device_id") = 0,
             py::arg("block_dim") = 0, py::arg("time_tile_samples") = 81920);
  module.def("dedisperse_multi_dm", &dedisperse_multi_dm_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm_low"),
             py::arg("dm_step"), py::arg("ndm"), py::arg("backend") = "cpu",
             py::arg("device_id") = 0, py::arg("block_dim") = 0,
             py::arg("time_tile_samples") = 81920);
  module.def("dedisperse_subband", &dedisperse_subband_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm_low"),
             py::arg("dm_step"), py::arg("ndm"), py::arg("backend") = "ascend",
             py::arg("subband_channels") = 32,
             py::arg("ndm_per_nominal") = 32, py::arg("device_id") = 0,
             py::arg("block_dim") = 0, py::arg("time_tile_samples") = 81920);
}

}  // namespace gaffa::python
