import unittest
from itertools import product

import numpy as np
from graphix.transpiler import Circuit

import tests.random_circuit as rc

SINGLE_QUBIT_GATE = ["I", "H", "S", "X", "Y", "Z", "RX", "RY", "RZ", "PASS"]
TWO_QUBIT_GATE = ["CNOT", "PASS"]


def generate_single_qubit_circuit(gate_instr):
    circuit = Circuit(1)
    if gate_instr == "I":
        circuit.i(0)
    elif gate_instr == "H":
        circuit.h(0)
    elif gate_instr == "S":
        circuit.s(0)
    elif gate_instr == "X":
        circuit.x(0)
    elif gate_instr == "Y":
        circuit.y(0)
    elif gate_instr == "Z":
        circuit.z(0)
    elif gate_instr == "RX":
        theta = np.random.rand() * 2 * np.pi
        circuit.rx(0, theta)
    elif gate_instr == "RY":
        theta = np.random.rand() * 2 * np.pi
        circuit.ry(0, theta)
    elif gate_instr == "RZ":
        theta = np.random.rand() * 2 * np.pi
        circuit.rz(0, theta)

    return circuit


def generate_two_qubit_circuit(gate_instr):
    circuit = Circuit(2)
    if gate_instr == "CNOT":
        circuit.cnot(0, 1)
    elif gate_instr == "PASS":
        pass

    return circuit


def generate_gate_combination_circuit(gate_instrs):
    circuit = Circuit(2)
    for gate_instr in gate_instrs:
        if gate_instr == "I":
            circuit.i(0)
        elif gate_instr == "H":
            circuit.h(0)
        elif gate_instr == "S":
            circuit.s(0)
        elif gate_instr == "X":
            circuit.x(0)
        elif gate_instr == "Y":
            circuit.y(0)
        elif gate_instr == "Z":
            circuit.z(0)
        elif gate_instr == "RX":
            theta = np.random.rand() * 2 * np.pi
            circuit.rx(0, theta)
        elif gate_instr == "RY":
            theta = np.random.rand() * 2 * np.pi
            circuit.ry(0, theta)
        elif gate_instr == "RZ":
            theta = np.random.rand() * 2 * np.pi
            circuit.rz(0, theta)
        elif gate_instr == "CNOT":
            circuit.cnot(0, 1)
        elif gate_instr == "PASS":
            pass

    return circuit


class TestPattern(unittest.TestCase):
    def test_elementary_gate(self):
        global SINGLE_QUBIT_GATE
        for gate_instr in SINGLE_QUBIT_GATE:
            with self.subTest(gate=gate_instr):
                circuit = generate_single_qubit_circuit(gate_instr)
                mbqcgraph = circuit.to_mbqcgraph()
                mbqcgraph.update_flow()
                sv_mbqc = mbqcgraph.simulate_mbqc()

                state = circuit.simulate_statevector()
                np.testing.assert_almost_equal(np.abs(np.dot(sv_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_two_qubit_gate(self):
        global TWO_QUBIT_GATE
        for gate_instr in TWO_QUBIT_GATE:
            with self.subTest(gate=gate_instr):
                circuit = generate_two_qubit_circuit(gate_instr)
                mbqcgraph = circuit.to_mbqcgraph()
                mbqcgraph.update_flow()
                sv_mbqc = mbqcgraph.simulate_mbqc()

                state = circuit.simulate_statevector()
                np.testing.assert_almost_equal(np.abs(np.dot(sv_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_gate_combination(self):
        global SINGLE_QUBIT_GATE, TWO_QUBIT_GATE
        for gate_instr1, gate_instr2, gate_instr3 in product(SINGLE_QUBIT_GATE, TWO_QUBIT_GATE, SINGLE_QUBIT_GATE):
            with self.subTest(
                gate_instr1=gate_instr1,
                gate_instr2=gate_instr2,
                gate_instr3=gate_instr3,
            ):
                circuit = generate_gate_combination_circuit([gate_instr1, gate_instr2, gate_instr3])
                mbqcgraph = circuit.to_mbqcgraph()
                mbqcgraph.update_flow()
                sv_mbqc = mbqcgraph.simulate_mbqc()

                state = circuit.simulate_statevector()
                np.testing.assert_almost_equal(np.abs(np.dot(sv_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_random_circuit(self):
        circuit = rc.get_rand_circuit(5, 5)
        mbqcgraph = circuit.to_mbqcgraph()
        mbqcgraph.update_flow()
        pattern = mbqcgraph.get_pattern()
        pattern.minimize_space()
        sv_mbqc = pattern.simulate_pattern()

        state = circuit.simulate_statevector()
        np.testing.assert_almost_equal(np.abs(np.dot(sv_mbqc.flatten().conjugate(), state.flatten())), 1)


if __name__ == "__main__":
    unittest.main()
