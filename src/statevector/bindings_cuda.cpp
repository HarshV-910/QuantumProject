
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "statevector_cuda.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_statevector_cuda, m) {
    m.doc() = "CUDA statevector simulator backend";

    py::class_<CudaStateVector>(m, "CudaStateVector")
        .def(py::init<>())

        .def(
            "initialize",
            &CudaStateVector::initialize,
            py::arg("n")
        )

        .def(
            "n_qubits",
            &CudaStateVector::n_qubits
        )

        .def(
            "apply_1q_gate",
            &CudaStateVector::apply_1q_gate,
            py::arg("gate"),
            py::arg("qubit"),
            py::arg("theta") = 0.0,
            py::arg("phi") = 0.0,
            py::arg("lam") = 0.0
        )

        .def(
            "apply_2q_gate",
            &CudaStateVector::apply_2q_gate,
            py::arg("gate"),
            py::arg("ctrl"),
            py::arg("tgt"),
            py::arg("theta") = 0.0,
            py::arg("phi") = 0.0,
            py::arg("lam") = 0.0
        )

        .def(
            "measure",
            &CudaStateVector::measure,
            py::arg("qubit")
        )

        .def(
            "sample",
            &CudaStateVector::sample,
            py::arg("shots")
        )

        .def(
            "expectation",
            &CudaStateVector::expectation,
            py::arg("pauli_string")
        )

        .def(
            "statevector",
            &CudaStateVector::statevector
        );
}
