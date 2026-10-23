"""Focused tests for the weighted DD implementation. Run: python -m unittest tests_dd.py"""
import math
import unittest
import numpy as np

from backends.dd import DDSimulator


def dense_apply_1q(state, matrix, q):
    out = state.copy()
    stride = 1 << q
    for start in range(0, len(state), 2 * stride):
        a = state[start:start + stride].copy()
        b = state[start + stride:start + 2 * stride].copy()
        out[start:start + stride] = matrix[0, 0] * a + matrix[0, 1] * b
        out[start + stride:start + 2 * stride] = matrix[1, 0] * a + matrix[1, 1] * b
    return out


def dense_apply_2q(state, matrix, ctrl, tgt):
    out = np.zeros_like(state)
    for i, amp in enumerate(state):
        cb, tb = (i >> ctrl) & 1, (i >> tgt) & 1
        col = 2 * cb + tb
        for oc in (0, 1):
            for ot in (0, 1):
                j = (i & ~(1 << ctrl) & ~(1 << tgt)) | (oc << ctrl) | (ot << tgt)
                out[j] += matrix[2 * oc + ot, col] * amp
    return out


class WeightedDDTests(unittest.TestCase):
    def test_initial_state_and_h(self):
        sim = DDSimulator()
        sim.initialize(3)
        sim.apply_1q_gate("H", 1)
        expected = np.zeros(8, complex)
        expected[0] = expected[2] = 1 / math.sqrt(2)
        np.testing.assert_allclose(sim.statevector, expected, atol=1e-10)

    def test_controlled_gates_and_swaps(self):
        sim = DDSimulator()
        sim.initialize(4)
        state = np.zeros(16, complex)
        state[0] = 1
        gates = {
            "CNOT": np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex),
            "CZ": np.diag([1,1,1,-1]).astype(complex),
            "SWAP": np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], complex),
        }
        rng = np.random.default_rng(42)
        oneq = {
            "H": np.array([[1,1],[1,-1]], complex) / math.sqrt(2),
            "T": np.diag([1, np.exp(1j * math.pi / 4)]).astype(complex),
            "Y": np.array([[0,-1j],[1j,0]], complex),
        }
        for _ in range(18):
            if rng.random() < .55:
                q = int(rng.integers(4))
                gate = list(oneq)[int(rng.integers(len(oneq)))]
                sim.apply_1q_gate(gate, q)
                state = dense_apply_1q(state, oneq[gate], q)
            else:
                c, t = map(int, rng.choice(4, 2, replace=False))
                gate = list(gates)[int(rng.integers(len(gates)))]
                sim.apply_2q_gate(gate, c, t)
                state = dense_apply_2q(state, gates[gate], c, t)
            np.testing.assert_allclose(sim.statevector, state, atol=1e-9)

    def test_controlled_u(self):
        sim = DDSimulator()
        sim.initialize(3)
        sim.apply_1q_gate("H", 0)
        sim.apply_1q_gate("H", 2)
        p = dict(theta=.37, phi=-.2, lam=.8)
        sim.apply_2q_gate("CU", 2, 0, **p)
        u = sim._u3(**p)
        m = np.eye(4, dtype=complex)
        m[2:4, 2:4] = u
        state = np.zeros(8, complex)
        state[0] = 1
        state = dense_apply_1q(state, np.array([[1,1],[1,-1]], complex)/math.sqrt(2), 0)
        state = dense_apply_1q(state, np.array([[1,1],[1,-1]], complex)/math.sqrt(2), 2)
        state = dense_apply_2q(state, m, 2, 0)
        np.testing.assert_allclose(sim.statevector, state, atol=1e-9)

    def test_expectations(self):
        sim = DDSimulator()
        sim.initialize(2)
        sim.apply_1q_gate("H", 0)
        self.assertAlmostEqual(sim.expectation("ZI"), 0.0, places=9)
        self.assertAlmostEqual(sim.expectation("XI"), 1.0, places=9)
        self.assertAlmostEqual(sim.expectation("IZ"), 1.0, places=9)

    def test_measurement_collapse(self):
        sim = DDSimulator()
        sim.initialize(2)
        sim.apply_1q_gate("H", 0)
        result = sim.measure(0)
        self.assertIn(result, (0, 1))
        self.assertAlmostEqual(sim.expectation("IZ"), 1.0, places=9)
        counts = sim.sample(20)
        self.assertEqual(sum(counts.values()), 20)
        self.assertTrue(all(k[0] == str(result) for k in counts))

if __name__ == "__main__":
    unittest.main()
