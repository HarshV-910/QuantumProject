/*low-level performance kernels designed to accelerate tensor operations for an MPS quantum simulator.
Exposed via C linkage (extern "C"), these functions perform the heavy mathematical lifting for
1-qubit and adjacent 2-qubit gate operations
using standard 1D array row-major indexing.*/

#include <algorithm>
#include <complex>
#include <new>
#include <complex>
using namespace std;

// one-qubit MPS tensor is stored as [left_bond, physical_bit, right_bond].
extern "C"
{

    int mps_apply_1q(complex<double> *tensor, int left_dim, int right_dim, const complex<double> *gate);

    // Contract neighbouring tensors, apply a 4x4 gate, and return the merged tensor [left_bond, output_left_bit, output_right_bit, right_bond].
    int mps_apply_adjacent_2q(const complex<double> *left_tensor,
                              int left_bond, int shared_bond,
                              const complex<double> *right_tensor,
                              int right_bond,
                              const complex<double> *gate,
                              complex<double> *merged_tensor);
}

extern "C" int mps_apply_1q(complex<double> *tensor, int left_dim, int right_dim, const complex<double> *gate)
{
    if (!tensor || !gate || left_dim < 1 || right_dim < 1)
        return 1;

    // Work into a temporary buffer so input and output may safely overlap.
    const int count = left_dim * 2 * right_dim;
    auto *output = new (nothrow) complex<double>[count];
    if (!output)
        return 2;

    for (int l = 0; l < left_dim; ++l)
    {
        for (int out_bit = 0; out_bit < 2; ++out_bit)
        {
            for (int r = 0; r < right_dim; ++r)
            {
                complex<double> value{0.0, 0.0};
                for (int in_bit = 0; in_bit < 2; ++in_bit)
                {
                    const int source = (l * 2 + in_bit) * right_dim + r;
                    value += gate[out_bit * 2 + in_bit] * tensor[source];
                }
                output[(l * 2 + out_bit) * right_dim + r] = value;
            }
        }
    }
    copy(output, output + count, tensor);
    delete[] output;
    return 0;
}

extern "C" int mps_apply_adjacent_2q(
    const complex<double> *left_tensor,
    int left_bond, int shared_bond,
    const complex<double> *right_tensor,
    int right_bond,
    const complex<double> *gate,
    complex<double> *merged_tensor)
{
    if (!left_tensor || !right_tensor || !gate || !merged_tensor ||
        left_bond < 1 || shared_bond < 1 || right_bond < 1)
        return 1;

    // First contract the shared bond, then apply the two-site gate. The merged
    // tensor is [left_bond, physical-left, physical-right, right_bond].
    for (int l = 0; l < left_bond; ++l)
    {
        for (int out_a = 0; out_a < 2; ++out_a)
        {
            for (int out_b = 0; out_b < 2; ++out_b)
            {
                for (int r = 0; r < right_bond; ++r)
                {
                    complex<double> value{0.0, 0.0};
                    const int output_basis = out_a * 2 + out_b;
                    for (int in_a = 0; in_a < 2; ++in_a)
                    {
                        for (int in_b = 0; in_b < 2; ++in_b)
                        {
                            const int input_basis = in_a * 2 + in_b;
                            const auto g = gate[output_basis * 4 + input_basis];
                            if (g == complex<double>{0.0, 0.0})
                                continue;
                            for (int s = 0; s < shared_bond; ++s)
                            {
                                const int ia = (l * 2 + in_a) * shared_bond + s;
                                const int ib = (s * 2 + in_b) * right_bond + r;
                                value += g * left_tensor[ia] * right_tensor[ib];
                            }
                        }
                    }
                    const int out_index = ((l * 2 + out_a) * 2 + out_b) * right_bond + r;
                    merged_tensor[out_index] = value;
                }
            }
        }
    }
    return 0;
}
