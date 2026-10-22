
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "statevector.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_statevector, m) {
    m.doc() = "C++ state-vector simulation kernel";

    py::class_<StateVector>(m, "StateVector")
        .def(py::init<>())

        .def("initialize", &StateVector::initialize)
        .def("n_qubits", &StateVector::n_qubits)

        .def(
            "apply_1q_gate",
            &StateVector::apply_1q_gate,
            py::arg("gate"),
            py::arg("qubit"),
            py::arg("theta") = 0.0,
            py::arg("phi") = 0.0,
            py::arg("lam") = 0.0
        )

        .def(
            "apply_2q_gate",
            &StateVector::apply_2q_gate,
            py::arg("gate"),
            py::arg("ctrl"),
            py::arg("tgt"),
            py::arg("theta") = 0.0,
            py::arg("phi") = 0.0,
            py::arg("lam") = 0.0
        )

        .def("measure", &StateVector::measure)
        .def("statevector", &StateVector::statevector);
}
