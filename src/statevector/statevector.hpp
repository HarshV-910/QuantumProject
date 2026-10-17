
#ifndef STATEVECTOR_HPP
#define STATEVECTOR_HPP

#include <complex>
#include <string>
#include <vector>

class StateVector {
public:
    using Complex = std::complex<double>;

    StateVector();

    void initialize(int n);
    int n_qubits() const;

    void apply_1q_gate(
        const std::string& gate,
        int qubit,
        double theta = 0.0,
        double phi = 0.0,
        double lam = 0.0
    );

    void apply_2q_gate(
        const std::string& gate,
        int ctrl,
        int tgt,
        double theta = 0.0,
        double phi = 0.0,
        double lam = 0.0
    );

    int measure(int qubit);
    std::vector<Complex> statevector() const;

private:
    int n_;
    std::vector<Complex> state_;

    void validate_qubit(int qubit) const;
    void apply_controlled_u(
        int ctrl,
        int tgt,
        double theta,
        double phi,
        double lam
    );
};

#endif
