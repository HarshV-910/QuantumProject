import numpy as np
from backends.statevector import StatevectorSimulator


def test_initialization():
    sim = StatevectorSimulator()
    sim.initialize(2)

    assert sim.n_qubits == 2
    assert np.allclose(sim._state, [1, 0, 0, 0])


def test_hadamard():
    sim = StatevectorSimulator()
    sim.initialize(1)

    sim.apply_1q_gate("H", 0)

    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(sim._state, expected)


def test_x_gate():
    sim = StatevectorSimulator()
    sim.initialize(1)

    sim.apply_1q_gate("X", 0)

    assert np.allclose(sim._state, [0, 1])


def test_cnot():
    sim = StatevectorSimulator()
    sim.initialize(2)

    sim.apply_1q_gate("X", 0)
    sim.apply_2q_gate("CNOT", 0, 1)

    assert np.allclose(sim._state, [0, 0, 0, 1])


def test_expectation():
    sim = StatevectorSimulator()
    sim.initialize(1)

    assert np.isclose(sim.expectation("Z"), 1.0)

    sim.apply_1q_gate("X", 0)

    assert np.isclose(sim.expectation("Z"), -1.0)


def test_sampling():
    sim = StatevectorSimulator()
    sim.initialize(1)

    sim.apply_1q_gate("H", 0)

    original_state = sim._state.copy()
    counts = sim.sample(1000)

    assert sum(counts.values()) == 1000
    assert set(counts).issubset({"0", "1"})
    assert np.allclose(sim._state, original_state)

    assert 350 < counts.get("0", 0) < 650
    assert 350 < counts.get("1", 0) < 650


if __name__ == "__main__":
    test_initialization()
    test_hadamard()
    test_x_gate()
    test_cnot()
    test_expectation()
    test_sampling()

    print("All Member A basic tests passed.")