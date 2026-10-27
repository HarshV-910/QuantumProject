#pragma once

#include <cuComplex.h>
#include <complex>
#include <string>
#include <vector>

class CudaStateVector {
public:
    CudaStateVector();
    ~CudaStateVector();
    CudaStateVector(const CudaStateVector&) = delete;
    CudaStateVector& operator=(const CudaStateVector&) = delete;

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
    std::vector<std::string> sample(int shots);
    double expectation(const std::string& pauli_string) const;

    std::vector<std::complex<double>> statevector() const;

private:
    int n_;
    int dim_;

    cuDoubleComplex* state_;
    cuDoubleComplex* scratch_;

    void check_initialized() const;
    void check_qubit(int q) const;

    std::vector<std::complex<double>> host_state() const;

    void set_host_state(
        const std::vector<std::complex<double>>& values
    );
};