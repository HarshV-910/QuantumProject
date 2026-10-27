
#include "statevector_cuda.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__global__ void oneq_kernel(
    const cuDoubleComplex* input,
    cuDoubleComplex* output,
    int dim,
    int qubit,
    cuDoubleComplex a,
    cuDoubleComplex b,
    cuDoubleComplex c,
    cuDoubleComplex d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    int mask = 1 << qubit;
    int j = i ^ mask;

    if ((i & mask) == 0) {
        output[i] = cuCadd(
            cuCmul(a, input[i]),
            cuCmul(b, input[j])
        );
    } else {
        output[i] = cuCadd(
            cuCmul(c, input[j]),
            cuCmul(d, input[i])
        );
    }
}

__global__ void twoq_kernel(
    const cuDoubleComplex* input,
    cuDoubleComplex* output,
    int dim,
    int ctrl,
    int tgt,
    int gate,
    cuDoubleComplex a,
    cuDoubleComplex b,
    cuDoubleComplex c,
    cuDoubleComplex d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    int cmask = 1 << ctrl;
    int tmask = 1 << tgt;

    bool control_bit = (i & cmask) != 0;
    bool target_bit = (i & tmask) != 0;

    if (gate == 0) {
        output[i] = input[
            control_bit ? (i ^ tmask) : i
        ];
        return;
    }

    if (gate == 1) {
        cuDoubleComplex value = input[i];

        if (control_bit && target_bit) {
            value = make_cuDoubleComplex(
                -cuCreal(value),
                -cuCimag(value)
            );
        }

        output[i] = value;
        return;
    }

    if (gate == 2) {
        int source = (control_bit != target_bit)
            ? (i ^ cmask ^ tmask)
            : i;

        output[i] = input[source];
        return;
    }

    if (!control_bit) {
        output[i] = input[i];
        return;
    }

    int j = i ^ tmask;

    if (!target_bit) {
        output[i] = cuCadd(
            cuCmul(a, input[i]),
            cuCmul(b, input[j])
        );
    } else {
        output[i] = cuCadd(
            cuCmul(c, input[j]),
            cuCmul(d, input[i])
        );
    }
}

void make_u3(
    double theta,
    double phi,
    double lam,
    cuDoubleComplex& a,
    cuDoubleComplex& b,
    cuDoubleComplex& c,
    cuDoubleComplex& d
) {
    double cs = std::cos(theta / 2.0);
    double sn = std::sin(theta / 2.0);

    a = make_cuDoubleComplex(cs, 0.0);

    b = make_cuDoubleComplex(
        -std::cos(lam) * sn,
        -std::sin(lam) * sn
    );

    c = make_cuDoubleComplex(
        std::cos(phi) * sn,
        std::sin(phi) * sn
    );

    d = make_cuDoubleComplex(
        std::cos(phi + lam) * cs,
        std::sin(phi + lam) * cs
    );
}

} 

CudaStateVector::CudaStateVector()
    : n_(0), dim_(0), state_(nullptr), scratch_(nullptr) {}

CudaStateVector::~CudaStateVector() {
    if (state_) cudaFree(state_);
    if (scratch_) cudaFree(scratch_);
}

void CudaStateVector::check_initialized() const {
    if (!state_ || !scratch_ || dim_ <= 0) {
        throw std::runtime_error("Simulator is not initialized");
    }
}

void CudaStateVector::check_qubit(int q) const {
    check_initialized();

    if (q < 0 || q >= n_) {
        throw std::out_of_range("Invalid qubit index");
    }
}

void CudaStateVector::initialize(int n) {
    if (n < 1 || n >= 31) {
        throw std::invalid_argument(
            "Number of qubits must be between 1 and 30"
        );
    }

    if (state_) {
        cuda_check(cudaFree(state_));
        state_ = nullptr;
    }

    if (scratch_) {
        cuda_check(cudaFree(scratch_));
        scratch_ = nullptr;
    }

    n_ = n;
    dim_ = 1 << n_;

    size_t bytes =
        static_cast<size_t>(dim_) * sizeof(cuDoubleComplex);

    try {
        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&state_), bytes
        ));

        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&scratch_), bytes
        ));

        cuda_check(cudaMemset(state_, 0, bytes));
        cuda_check(cudaMemset(scratch_, 0, bytes));

        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

        cuda_check(cudaMemcpy(
            state_,
            &one,
            sizeof(one),
            cudaMemcpyHostToDevice
        ));
    } catch (...) {
        if (state_) cudaFree(state_);
        if (scratch_) cudaFree(scratch_);

        state_ = nullptr;
        scratch_ = nullptr;
        n_ = 0;
        dim_ = 0;

        throw;
    }
}

int CudaStateVector::n_qubits() const {
    check_initialized();
    return n_;
}

void CudaStateVector::apply_1q_gate(
    const std::string& gate,
    int qubit,
    double theta,
    double phi,
    double lam
) {
    check_qubit(qubit);

    cuDoubleComplex a, b, c, d;

    if (gate == "H") {
        double r = 1.0 / std::sqrt(2.0);

        a = make_cuDoubleComplex(r, 0);
        b = make_cuDoubleComplex(r, 0);
        c = make_cuDoubleComplex(r, 0);
        d = make_cuDoubleComplex(-r, 0);
    } else if (gate == "X") {
        a = make_cuDoubleComplex(0, 0);
        b = make_cuDoubleComplex(1, 0);
        c = make_cuDoubleComplex(1, 0);
        d = make_cuDoubleComplex(0, 0);
    } else if (gate == "Y") {
        a = make_cuDoubleComplex(0, 0);
        b = make_cuDoubleComplex(0, -1);
        c = make_cuDoubleComplex(0, 1);
        d = make_cuDoubleComplex(0, 0);
    } else if (gate == "Z") {
        a = make_cuDoubleComplex(1, 0);
        b = make_cuDoubleComplex(0, 0);
        c = make_cuDoubleComplex(0, 0);
        d = make_cuDoubleComplex(-1, 0);
    } else if (gate == "S") {
        a = make_cuDoubleComplex(1, 0);
        b = make_cuDoubleComplex(0, 0);
        c = make_cuDoubleComplex(0, 0);
        d = make_cuDoubleComplex(0, 1);
    } else if (gate == "T") {
        double r = 1.0 / std::sqrt(2.0);

        a = make_cuDoubleComplex(1, 0);
        b = make_cuDoubleComplex(0, 0);
        c = make_cuDoubleComplex(0, 0);
        d = make_cuDoubleComplex(r, r);
    } else if (gate == "U") {
        make_u3(theta, phi, lam, a, b, c, d);
    } else {
        throw std::invalid_argument(
            "Unsupported single-qubit gate: " + gate
        );
    }

    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;

    oneq_kernel<<<blocks, threads>>>(
        state_, scratch_, dim_, qubit, a, b, c, d
    );

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    std::swap(state_, scratch_);
}

void CudaStateVector::apply_2q_gate(
    const std::string& gate,
    int ctrl,
    int tgt,
    double theta,
    double phi,
    double lam
) {
    check_qubit(ctrl);
    check_qubit(tgt);

    if (ctrl == tgt) {
        throw std::invalid_argument(
            "Control and target qubits must differ"
        );
    }

    int gate_id;

    if (gate == "CNOT") {
        gate_id = 0;
    } else if (gate == "CZ") {
        gate_id = 1;
    } else if (gate == "SWAP") {
        gate_id = 2;
    } else if (gate == "CU") {
        gate_id = 3;
    } else {
        throw std::invalid_argument(
            "Unsupported two-qubit gate: " + gate
        );
    }

    cuDoubleComplex a = make_cuDoubleComplex(1, 0);
    cuDoubleComplex b = make_cuDoubleComplex(0, 0);
    cuDoubleComplex c = make_cuDoubleComplex(0, 0);
    cuDoubleComplex d = make_cuDoubleComplex(1, 0);

    if (gate_id == 3) {
        make_u3(theta, phi, lam, a, b, c, d);
    }

    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;

    twoq_kernel<<<blocks, threads>>>(
        state_, scratch_, dim_, ctrl, tgt, gate_id,
        a, b, c, d
    );

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    std::swap(state_, scratch_);
}

std::vector<std::complex<double>>
CudaStateVector::host_state() const {
    check_initialized();

    std::vector<cuDoubleComplex> raw(dim_);

    cuda_check(cudaMemcpy(
        raw.data(),
        state_,
        static_cast<size_t>(dim_) * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost
    ));

    std::vector<std::complex<double>> result(dim_);

    for (int i = 0; i < dim_; ++i) {
        result[i] = {
            cuCreal(raw[i]),
            cuCimag(raw[i])
        };
    }

    return result;
}

void CudaStateVector::set_host_state(
    const std::vector<std::complex<double>>& values
) {
    check_initialized();

    if (static_cast<int>(values.size()) != dim_) {
        throw std::invalid_argument(
            "Incorrect statevector dimension"
        );
    }

    std::vector<cuDoubleComplex> raw(dim_);

    for (int i = 0; i < dim_; ++i) {
        raw[i] = make_cuDoubleComplex(
            values[i].real(),
            values[i].imag()
        );
    }

    cuda_check(cudaMemcpy(
        state_,
        raw.data(),
        static_cast<size_t>(dim_) * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice
    ));
}

std::vector<std::complex<double>>
CudaStateVector::statevector() const {
    return host_state();
}

int CudaStateVector::measure(int qubit) {
    check_qubit(qubit);

    auto values = host_state();
    int mask = 1 << qubit;
    double p1 = 0.0;

    for (int i = 0; i < dim_; ++i) {
        if (i & mask) {
            p1 += std::norm(values[i]);
        }
    }

    p1 = std::clamp(p1, 0.0, 1.0);

    static thread_local std::mt19937_64 rng(
        std::random_device{}()
    );

    int bit = std::bernoulli_distribution(p1)(rng);
    double probability = bit ? p1 : (1.0 - p1);

    if (probability <= 0.0) {
        throw std::runtime_error(
            "Impossible measurement outcome"
        );
    }

    double scale = 1.0 / std::sqrt(probability);

    for (int i = 0; i < dim_; ++i) {
        bool current_bit = (i & mask) != 0;

        if (current_bit != (bit != 0)) {
            values[i] = {0.0, 0.0};
        } else {
            values[i] *= scale;
        }
    }

    set_host_state(values);
    return bit;
}

std::vector<std::string>
CudaStateVector::sample(int shots) {
    check_initialized();

    if (shots < 0) {
        throw std::invalid_argument(
            "shots must be non-negative"
        );
    }

    auto values = host_state();

    std::vector<double> probabilities(dim_);
    double total = 0.0;

    for (int i = 0; i < dim_; ++i) {
        probabilities[i] = std::norm(values[i]);
        total += probabilities[i];
    }

    if (total <= 0.0) {
        throw std::runtime_error(
            "Statevector has zero norm"
        );
    }

    for (double& p : probabilities) {
        p /= total;
    }

    static thread_local std::mt19937_64 rng(
        std::random_device{}()
    );
    std::discrete_distribution<int> distribution(
        probabilities.begin(), probabilities.end()
    );

    std::vector<std::string> result;
    result.reserve(shots);

    for (int s = 0; s < shots; ++s) {
        int index = distribution(rng);
        std::string bits(n_, '0');
        
        //We have kept qubit-0 to the left
        for (int q = 0; q < n_; ++q) {
            if (index & (1 << q)) {
                bits[q] = '1';
            }
        }

        result.push_back(bits);
    }

    return result;
}

double CudaStateVector::expectation(
    const std::string& pauli_string
) const {
    check_initialized();

    if (static_cast<int>(pauli_string.size()) != n_) {
        throw std::invalid_argument(
            "Pauli string length mismatch"
        );
    }

    auto values = host_state();
    std::complex<double> result(0.0, 0.0);

    for (int i = 0; i < dim_; ++i) {
        int j = i;
        std::complex<double> phase(1.0, 0.0);

        for (int q = 0; q < n_; ++q) {
            // rightmost Pauli character maps to qubit 0.
            char p = pauli_string[n_ - 1 - q];
            bool bit = (i & (1 << q)) != 0;

            if (p == 'I') {
                continue;
            } else if (p == 'X') {
                j ^= (1 << q);
            } else if (p == 'Y') {
                j ^= (1 << q);
                phase *= bit
                    ? std::complex<double>(0.0, -1.0)
                    : std::complex<double>(0.0, 1.0);
            } else if (p == 'Z') {
                if (bit) phase = -phase;
            } else {
                throw std::invalid_argument(
                    "Pauli string must contain only I, X, Y, Z"
                );
            }
        }

        result += std::conj(values[j]) * phase * values[i];
    }

    if (std::abs(result.imag()) > 1e-9) {
        throw std::runtime_error(
            "Expectation has imaginary residue"
        );
    }

    return result.real();
}
