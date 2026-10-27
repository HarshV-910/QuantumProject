
#include "statevector.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <algorithm>

namespace {
    constexpr double PI = 3.14159265358979323846;

    std::mt19937_64& rng() {
        static thread_local std::mt19937_64 engine(
            std::random_device{}()
        );
        return engine;
    }
}

StateVector::StateVector() : n_(0) {}

void StateVector::initialize(int n) {
    if (n < 1 || n >= 63) {
        throw std::invalid_argument("Invalid number of qubits");
    }

    n_ = n;
    state_.assign(std::size_t{1} << n, Complex{0.0, 0.0});
    state_[0] = Complex{1.0, 0.0};
}

int StateVector::n_qubits() const {
    if (n_ == 0) {
        throw std::runtime_error("Simulator is not initialized");
    }
    return n_;
}

void StateVector::validate_qubit(int qubit) const {
    n_qubits();

    if (qubit < 0 || qubit >= n_) {
        throw std::out_of_range("Invalid qubit index");
    }
}

void StateVector::apply_1q_gate(
    const std::string& gate,
    int qubit,
    double theta,
    double phi,
    double lam
) {
    validate_qubit(qubit);

    const std::size_t mask = std::size_t{1} << qubit;

    Complex u00, u01, u10, u11;
    const Complex I{0.0, 1.0};

    if (gate == "H") {
        const double s = 1.0 / std::sqrt(2.0);
        u00 = s;  u01 = s;
        u10 = s;  u11 = -s;
    } else if (gate == "X") {
        u00 = 0; u01 = 1;
        u10 = 1; u11 = 0;
    } else if (gate == "Y") {
        u00 = 0;  u01 = -I;
        u10 = I;  u11 = 0;
    } else if (gate == "Z") {
        u00 = 1; u01 = 0;
        u10 = 0; u11 = -1;
    } else if (gate == "S") {
        u00 = 1; u01 = 0;
        u10 = 0; u11 = I;
    } else if (gate == "T") {
        u00 = 1; u01 = 0;
        u10 = 0; u11 = std::exp(I * PI / 4.0);
    } else if (gate == "U") {
        u00 = std::cos(theta / 2.0);
        u01 = -std::exp(I * lam) * std::sin(theta / 2.0);
        u10 = std::exp(I * phi) * std::sin(theta / 2.0);
        u11 = std::exp(I * (phi + lam)) *
              std::cos(theta / 2.0);
    } else {
        throw std::invalid_argument("Unsupported one-qubit gate: " + gate);
    }

    for (std::size_t i = 0; i < state_.size(); ++i) {
        if ((i & mask) != 0) {
            continue;
        }

        const std::size_t j = i | mask;
        const Complex a = state_[i];
        const Complex b = state_[j];

        state_[i] = u00 * a + u01 * b;
        state_[j] = u10 * a + u11 * b;
    }
}

void StateVector::apply_controlled_u(
    int ctrl,
    int tgt,
    double theta,
    double phi,
    double lam
) {
    const std::size_t cm = std::size_t{1} << ctrl;
    const std::size_t tm = std::size_t{1} << tgt;
    const Complex I{0.0, 1.0};

    const Complex u00 = std::cos(theta / 2.0);
    const Complex u01 =
        -std::exp(I * lam) * std::sin(theta / 2.0);
    const Complex u10 =
        std::exp(I * phi) * std::sin(theta / 2.0);
    const Complex u11 =
        std::exp(I * (phi + lam)) * std::cos(theta / 2.0);

    for (std::size_t i = 0; i < state_.size(); ++i) {
        if ((i & cm) == 0 || (i & tm) != 0) {
            continue;
        }

        const std::size_t j = i | tm;
        const Complex a = state_[i];
        const Complex b = state_[j];

        state_[i] = u00 * a + u01 * b;
        state_[j] = u10 * a + u11 * b;
    }
}

void StateVector::apply_2q_gate(
    const std::string& gate,
    int ctrl,
    int tgt,
    double theta,
    double phi,
    double lam
) {
    validate_qubit(ctrl);
    validate_qubit(tgt);

    if (ctrl == tgt) {
        throw std::invalid_argument("Control and target must differ");
    }

    const std::size_t cm = std::size_t{1} << ctrl;
    const std::size_t tm = std::size_t{1} << tgt;

    if (gate == "CNOT") {
        for (std::size_t i = 0; i < state_.size(); ++i) {
            if ((i & cm) != 0 && (i & tm) == 0) {
                std::swap(state_[i], state_[i | tm]);
            }
        }
    } else if (gate == "CZ") {
        for (std::size_t i = 0; i < state_.size(); ++i) {
            if ((i & cm) != 0 && (i & tm) != 0) {
                state_[i] = -state_[i];
            }
        }
    } else if (gate == "SWAP") {
        for (std::size_t i = 0; i < state_.size(); ++i) {
            if ((i & cm) == 0 && (i & tm) != 0) {
                const std::size_t j = i ^ cm ^ tm;
                std::swap(state_[i], state_[j]);
            }
        }
    } else if (gate == "CU") {
        apply_controlled_u(ctrl, tgt, theta, phi, lam);
    } else {
        throw std::invalid_argument("Unsupported two-qubit gate: " + gate);
    }
}

int StateVector::measure(int qubit) {
    validate_qubit(qubit);

    const std::size_t mask = std::size_t{1} << qubit;
    double p1 = 0.0;

    for (std::size_t i = 0; i < state_.size(); ++i) {
        if ((i & mask) != 0) {
            p1 += std::norm(state_[i]);
        }
    }

    p1 = std::max(0.0, std::min(1.0, p1));
    std::bernoulli_distribution distribution(p1);
    const int outcome = distribution(rng());

    const double probability = outcome ? p1 : 1.0 - p1;
    if (probability <= 0.0) {
        throw std::runtime_error("Invalid measurement probability");
    }

    const double scale = 1.0 / std::sqrt(probability);

    for (std::size_t i = 0; i < state_.size(); ++i) {
        const bool bit = (i & mask) != 0;
        if (bit != static_cast<bool>(outcome)) {
            state_[i] = 0.0;
        } else {
            state_[i] *= scale;
        }
    }

    return outcome;
}

std::vector<StateVector::Complex> StateVector::statevector() const {
    n_qubits();
    return state_;
}
