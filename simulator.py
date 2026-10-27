# COL8422 Quantum Computing Systems — Project Harness
# Abstract base class and factory for student quantum circuit simulators.

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any


class UnsupportedOperationError(Exception):
    """Raised when a simulator backend does not support the requested operation."""


class QuantumSimulator(ABC):
    """Abstract base class that all student simulator backends must subclass.

    Each backend must be placed in a file named ``backends/<name>.py`` and must
    define a single class whose name follows the convention below:

    =====================  ===========================  =======================
    backend string         module path                  class name
    =====================  ===========================  =======================
    ``'statevector'``      ``backends/statevector.py``  ``StatevectorSimulator``
    ``'stabilizer'``       ``backends/stabilizer.py``   ``StabilizerSimulator``
    ``'mps'``              ``backends/mps.py``           ``MPSSimulator``
    ``'dd'``               ``backends/dd.py``            ``DDSimulator``
    =====================  ===========================  =======================

    The concrete class must accept no constructor arguments; initialisation of
    the quantum state is done through :meth:`initialize`.
    """

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, n: int) -> None:
        """Initialise (or re-initialise) the simulator to the |0…0⟩ state.

        Parameters
        ----------
        n:
            Number of qubits.  Must be a positive integer.
        """

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_1q_gate(self, gate: str, qubit: int, **params: Any) -> None:
        """Apply a single-qubit gate to ``qubit``.

        Supported gates
        ---------------
        ``'H'``
            Hadamard.
        ``'X'``
            Pauli-X (NOT).
        ``'Y'``
            Pauli-Y.
        ``'Z'``
            Pauli-Z.
        ``'S'``
            Phase gate (S = diag(1, i)).
        ``'T'``
            T gate (T = diag(1, e^{iπ/4})).
        ``'U'``
            General single-qubit unitary.  Requires keyword arguments
            ``theta``, ``phi``, ``lam`` (all floats, in radians) defining
            the standard U3 gate::

                U(θ,φ,λ) = [[cos(θ/2),            -e^{iλ} sin(θ/2)    ],
                             [e^{iφ} sin(θ/2),     e^{i(φ+λ)} cos(θ/2)]]

        Parameters
        ----------
        gate:
            Gate identifier string (case-sensitive).
        qubit:
            Zero-based qubit index.
        **params:
            Gate-specific keyword arguments (see above).

        Raises
        ------
        UnsupportedOperationError
            If the backend does not support the requested gate.
        ValueError
            If ``gate`` is not recognised or required keyword arguments are
            missing.
        """

    @abstractmethod
    def apply_2q_gate(self, gate: str, ctrl: int, tgt: int, **params: Any) -> None:
        """Apply a two-qubit gate with control qubit ``ctrl`` and target ``tgt``.

        Supported gates
        ---------------
        ``'CNOT'``
            Controlled-NOT.
        ``'CZ'``
            Controlled-Z.
        ``'SWAP'``
            SWAP gate.
        ``'CU'``
            Controlled-U gate.  Requires keyword arguments ``theta``, ``phi``,
            ``lam`` (floats, radians) with the same U3 convention as
            :meth:`apply_1q_gate`.

        Parameters
        ----------
        gate:
            Gate identifier string (case-sensitive).
        ctrl:
            Zero-based index of the control qubit.
        tgt:
            Zero-based index of the target qubit.
        **params:
            Gate-specific keyword arguments.

        Raises
        ------
        UnsupportedOperationError
            If the backend does not support the requested gate.
        ValueError
            If ``gate`` is not recognised or required keyword arguments are
            missing.
        """

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @abstractmethod
    def measure(self, qubit: int) -> int:
        """Measure a single qubit in the computational basis, collapsing the state.

        Parameters
        ----------
        qubit:
            Zero-based qubit index.

        Returns
        -------
        int
            Measurement outcome: 0 or 1.
        """

    @abstractmethod
    def sample(self, shots: int) -> dict[str, int]:
        """Sample the current state ``shots`` times without collapsing it.

        Parameters
        ----------
        shots:
            Number of measurement shots.

        Returns
        -------
        dict[str, int]
            Mapping from bit-string (e.g. ``'0110'``) to count.  Only
            bit-strings with non-zero counts are included.  Bit ordering
            follows qubit 0 at position 0 (leftmost character).
        """

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    @abstractmethod
    def expectation(self, pauli_string: str) -> float:
        """Compute the expectation value ⟨ψ|P|ψ⟩ for a tensor-product Pauli.

        Parameters
        ----------
        pauli_string:
            A string of length ``n_qubits`` whose characters are drawn from
            ``{'I', 'X', 'Y', 'Z'}``.  For example, ``'ZIZ'`` represents
            Z⊗I⊗Z acting on a 3-qubit register.

        Returns
        -------
        float
            Real-valued expectation (imaginary part is zero for Hermitian
            operators and should be discarded or asserted negligible).

        Raises
        ------
        UnsupportedOperationError
            If the backend cannot compute expectation values (e.g. a
            sampling-only stabilizer backend).
        ValueError
            If ``pauli_string`` has incorrect length or contains invalid
            characters.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Number of qubits in the current register (set by :meth:`initialize`)."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Mapping from backend name to (module stem, class name).
_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "statevector": ("statevector", "StatevectorSimulator"),
    "stabilizer":  ("stabilizer",  "StabilizerSimulator"),
    "mps":         ("mps",         "MPSSimulator"),
    "dd":          ("dd",          "DDSimulator"),
}


def Simulator(backend: str) -> QuantumSimulator:
    """Instantiate a :class:`QuantumSimulator` for the requested backend.

    The factory discovers backend implementations by importing from the
    ``backends/`` sub-package that must live alongside this file.  The
    expected layout is::

        part1/
        ├── simulator.py          ← this file
        ├── circuits.py
        ├── harness.py
        ├── benchmark.py
        └── backends/
            ├── __init__.py       ← may be empty
            ├── statevector.py    → defines StatevectorSimulator(QuantumSimulator)
            ├── stabilizer.py     → defines StabilizerSimulator(QuantumSimulator)
            ├── mps.py            → defines MPSSimulator(QuantumSimulator)
            └── dd.py             → defines DDSimulator(QuantumSimulator)

    Each module must define exactly one class whose name matches the
    ``<Title>Simulator`` convention shown above.  The class must accept
    **no constructor arguments**; all state initialisation goes through
    :meth:`QuantumSimulator.initialize`.

    Parameters
    ----------
    backend:
        One of ``'statevector'``, ``'stabilizer'``, ``'mps'``, ``'dd'``.

    Returns
    -------
    QuantumSimulator
        A freshly constructed (uninitialised) instance of the backend class.

    Raises
    ------
    ValueError
        If ``backend`` is not a recognised backend name.
    ImportError
        If the corresponding module cannot be found or imported.  The error
        message will include the expected file path and class name so that
        students can quickly identify what is missing.
    """
    if backend not in _BACKEND_MAP:
        supported = ", ".join(f"'{k}'" for k in _BACKEND_MAP)
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: {supported}."
        )

    module_stem, class_name = _BACKEND_MAP[backend]
    module_path = f"backends.{module_stem}"

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Backend '{backend}' not found.  Expected a module at "
            f"'backends/{module_stem}.py' (importable as '{module_path}') "
            f"that defines a class named '{class_name}' subclassing "
            f"QuantumSimulator.  Original error: {exc}"
        ) from exc

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(
            f"Module '{module_path}' was imported successfully, but it does "
            f"not define the expected class '{class_name}'.  Please ensure "
            f"your backend file exports a class with exactly that name."
        )

    return cls()
