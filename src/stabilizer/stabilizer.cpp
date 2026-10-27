// COL8422 — Stabilizer backend, C++ kernel (Member B)
//
// Aaronson–Gottesman tableau algorithm ("Improved Simulation of Stabilizer
// Circuits", PRA 70, 052328, 2004), Section III.
//
// Tableau layout (0-indexed; the paper is 1-indexed):
//   rows 0 .. n-1     destabilizers R_1 .. R_n
//   rows n .. 2n-1    stabilizers   R_{n+1} .. R_{2n}
//   row  2n           scratch row   (used by deterministic measurement)
//
// Each row stores n x-bits, n z-bits and one phase bit r.
//   (x,z) = (0,0) I   (1,0) X   (1,1) Y   (0,1) Z      r = 1  <=>  sign is -1
//
// Bits are packed 64 per uint64_t word, row-major: row i, word w lives at
// x[i*W + w] where W = ceil(n/64).  Row-major makes rowsum a fast word-wise
// XOR (the expensive part of measurement); a gate on qubit a touches one
// word in each of the 2n rows, so gates are O(n).
//
// Exposed through a plain C ABI so Python can load it with ctypes
// (no pybind11 / Python headers needed on the evaluation machine).

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {

struct Tableau {
    int n;                       // number of qubits
    int W;                       // 64-bit words per row (per x or z block)
    std::vector<uint64_t> x;     // (2n+1) * W
    std::vector<uint64_t> z;     // (2n+1) * W
    std::vector<uint8_t>  r;     // 2n+1 phase bits

    explicit Tableau(int n_)
        : n(n_), W((n_ + 63) / 64),
          x(static_cast<size_t>(2 * n_ + 1) * W, 0),
          z(static_cast<size_t>(2 * n_ + 1) * W, 0),
          r(2 * n_ + 1, 0) {
        // Standard initial tableau for |0...0>:
        //   destabilizer i = X_i, stabilizer i = Z_i, all phases +.
        for (int i = 0; i < n; ++i) {
            set_x(i, i, 1);
            set_z(n + i, i, 1);
        }
    }

    inline uint64_t *xrow(int i) { return &x[static_cast<size_t>(i) * W]; }
    inline uint64_t *zrow(int i) { return &z[static_cast<size_t>(i) * W]; }

    inline int get_x(int i, int a) const {
        return (x[static_cast<size_t>(i) * W + (a >> 6)] >> (a & 63)) & 1ULL;
    }
    inline int get_z(int i, int a) const {
        return (z[static_cast<size_t>(i) * W + (a >> 6)] >> (a & 63)) & 1ULL;
    }
    inline void set_x(int i, int a, int v) {
        uint64_t &w = x[static_cast<size_t>(i) * W + (a >> 6)];
        uint64_t m = 1ULL << (a & 63);
        w = v ? (w | m) : (w & ~m);
    }
    inline void set_z(int i, int a, int v) {
        uint64_t &w = z[static_cast<size_t>(i) * W + (a >> 6)];
        uint64_t m = 1ULL << (a & 63);
        w = v ? (w | m) : (w & ~m);
    }

    // ---------------------------------------------------------------
    // rowsum(h, i):  R_h  <-  R_i * R_h   (paper, Section III)
    //
    // The phase exponent is  2 r_h + 2 r_i + sum_j g(x_ij,z_ij,x_hj,z_hj)
    // (mod 4), where g in {-1,0,+1} is the power of i picked up when the
    // single-qubit Paulis are multiplied.  Bit-parallel form of g:
    //   g = +1  for  X*Y, Y*Z, Z*X   (cyclic order)
    //   g = -1  for  X*Z, Y*X, Z*Y   (anti-cyclic order)
    //   g =  0  if either is I or both are equal
    // so  sum g = popcount(plus) - popcount(minus).
    // For commuting rows the total is 0 or 2 (mod 4)  ->  r_h = 0 or 1.
    // ---------------------------------------------------------------
    void rowsum(int h, int i) {
        uint64_t *xh = xrow(h), *zh = zrow(h);
        const uint64_t *xi = xrow(i), *zi = zrow(i);
        int sum = 2 * r[h] + 2 * r[i];
        for (int w = 0; w < W; ++w) {
            const uint64_t x1 = xi[w], z1 = zi[w];   // left factor  (row i)
            const uint64_t x2 = xh[w], z2 = zh[w];   // right factor (row h)
            const uint64_t isX1 = x1 & ~z1, isY1 = x1 & z1, isZ1 = ~x1 & z1;
            const uint64_t isX2 = x2 & ~z2, isY2 = x2 & z2, isZ2 = ~x2 & z2;
            const uint64_t plus  = (isX1 & isY2) | (isY1 & isZ2) | (isZ1 & isX2);
            const uint64_t minus = (isX1 & isZ2) | (isY1 & isX2) | (isZ1 & isY2);
            sum += __builtin_popcountll(plus) - __builtin_popcountll(minus);
            xh[w] = x2 ^ x1;
            zh[w] = z2 ^ z1;
        }
        sum %= 4;
        if (sum < 0) sum += 4;
        r[h] = (sum == 2) ? 1 : 0;   // sum is never 1 or 3 for commuting rows
    }

    void copy_row(int dst, int src) {
        std::memcpy(xrow(dst), xrow(src), sizeof(uint64_t) * W);
        std::memcpy(zrow(dst), zrow(src), sizeof(uint64_t) * W);
        r[dst] = r[src];
    }

    void clear_row(int i) {
        std::memset(xrow(i), 0, sizeof(uint64_t) * W);
        std::memset(zrow(i), 0, sizeof(uint64_t) * W);
        r[i] = 0;
    }

    // ---------------------------------------------------------------
    // Clifford gates: each loops over the 2n real rows -> O(n).
    // (The scratch row is never touched by gates: it is cleared before use.)
    // ---------------------------------------------------------------

    // Hadamard: X <-> Z, Y -> -Y.
    void h(int a) {
        const int wa = a >> 6; const uint64_t m = 1ULL << (a & 63);
        for (int i = 0; i < 2 * n; ++i) {
            uint64_t &xw = x[static_cast<size_t>(i) * W + wa];
            uint64_t &zw = z[static_cast<size_t>(i) * W + wa];
            const int xa = (xw & m) != 0, za = (zw & m) != 0;
            r[i] ^= (xa & za);
            if (xa != za) { xw ^= m; zw ^= m; }   // swap the two bits
        }
    }

    // Phase S = diag(1, i): X -> Y, Y -> -X, Z -> Z.
    void s(int a) {
        const int wa = a >> 6; const uint64_t m = 1ULL << (a & 63);
        for (int i = 0; i < 2 * n; ++i) {
            uint64_t &xw = x[static_cast<size_t>(i) * W + wa];
            uint64_t &zw = z[static_cast<size_t>(i) * W + wa];
            const int xa = (xw & m) != 0, za = (zw & m) != 0;
            r[i] ^= (xa & za);
            if (xa) zw ^= m;                      // z_a ^= x_a
        }
    }

    // Pauli gates only flip signs (a Pauli conjugating a Pauli gives +/- itself).
    // X: flips sign of rows with Z or Y on a  -> r ^= z_a
    // Z: flips sign of rows with X or Y on a  -> r ^= x_a
    // Y: flips sign of rows with X or Z on a  -> r ^= x_a ^ z_a
    void pauli(int a, int kind /*1=X, 2=Y, 3=Z*/) {
        for (int i = 0; i < 2 * n; ++i) {
            const int xa = get_x(i, a), za = get_z(i, a);
            if (kind == 1) r[i] ^= za;
            else if (kind == 3) r[i] ^= xa;
            else r[i] ^= (xa ^ za);
        }
    }

    // CNOT a -> b:  X_a -> X_a X_b,  Z_b -> Z_a Z_b.
    void cnot(int a, int b) {
        for (int i = 0; i < 2 * n; ++i) {
            const int xa = get_x(i, a), za = get_z(i, a);
            const int xb = get_x(i, b), zb = get_z(i, b);
            r[i] ^= xa & zb & (xb ^ za ^ 1);
            set_x(i, b, xb ^ xa);
            set_z(i, a, za ^ zb);
        }
    }

    // CZ a,b (symmetric):  X_a -> X_a Z_b,  X_b -> Z_a X_b.
    // Equivalent to H(b) CNOT(a,b) H(b); written directly for speed.
    void cz(int a, int b) {
        for (int i = 0; i < 2 * n; ++i) {
            const int xa = get_x(i, a), za = get_z(i, a);
            const int xb = get_x(i, b), zb = get_z(i, b);
            r[i] ^= xa & xb & (za ^ zb);
            set_z(i, a, za ^ xb);
            set_z(i, b, zb ^ xa);
        }
    }

    // SWAP: just exchange the two columns; no sign change.
    void swap(int a, int b) {
        for (int i = 0; i < 2 * n; ++i) {
            const int xa = get_x(i, a), za = get_z(i, a);
            const int xb = get_x(i, b), zb = get_z(i, b);
            set_x(i, a, xb); set_x(i, b, xa);
            set_z(i, a, zb); set_z(i, b, za);
        }
    }

    // Returns p (0-indexed stabilizer row, n <= p < 2n) with x_pa = 1, or -1.
    int find_anticommuting_stabilizer(int a) const {
        for (int p = n; p < 2 * n; ++p)
            if (get_x(p, a)) return p;
        return -1;
    }

    // ---------------------------------------------------------------
    // Measurement of qubit a in the Z basis (paper Section III).
    // rand_bit is the coin flip used if the outcome is random; it is
    // supplied by Python so that all randomness comes from one seeded RNG.
    // ---------------------------------------------------------------
    int measure(int a, int rand_bit) {
        const int p = find_anticommuting_stabilizer(a);
        if (p >= 0) {
            // Case I: random outcome.  Z_a anticommutes with stabilizer R_p.
            // 1. Multiply R_p into every other row that anticommutes with Z_a,
            //    so that afterwards R_p is the only such row.  Row p-n is skipped:
            //    it is overwritten in step 2 (and it anticommutes with R_p, so
            //    the product would have an imaginary phase).
            for (int i = 0; i < 2 * n; ++i)
                if (i != p && i != p - n && get_x(i, a)) rowsum(i, p);
            // 2. Old stabilizer R_p becomes the destabilizer of the new generator.
            copy_row(p - n, p);
            // 3. New stabilizer generator is  (-1)^outcome Z_a.
            clear_row(p);
            set_z(p, a, 1);
            r[p] = static_cast<uint8_t>(rand_bit & 1);
            return rand_bit & 1;
        }
        // Case II: deterministic.  +/- Z_a is in the stabilizer group:
        //   Z_a = +/- prod_{h : x_ha = 1} R_{h+n}  (h over destabilizers).
        // Accumulate that product in the scratch row and read its sign.
        const int s = 2 * n;
        clear_row(s);
        for (int i = 0; i < n; ++i)
            if (get_x(i, a)) rowsum(s, i + n);
        return r[s];
    }

    // Is a measurement of qubit a random?  O(n), does not change the state.
    int is_random(int a) const { return find_anticommuting_stabilizer(a) >= 0; }

    // ---------------------------------------------------------------
    // <psi| P |psi> for a Pauli P given as bit vectors (px, pz), sign +.
    // Returns +1, -1 or 0.  Does not change the state (uses scratch row).
    //   * If P anticommutes with any stabilizer generator, <P> = 0.
    //   * Otherwise P = +/- (product of stabilizers); by invariant (iii)/(iv)
    //     the product contains R_{h+n} iff P anticommutes with destabilizer R_h.
    // ---------------------------------------------------------------
    int symplectic_with(int row, const uint64_t *px, const uint64_t *pz) {
        const uint64_t *xr = xrow(row), *zr = zrow(row);
        int parity = 0;
        for (int w = 0; w < W; ++w)
            parity ^= __builtin_popcountll((xr[w] & pz[w]) ^ (zr[w] & px[w])) & 1;
        return parity;
    }

    int expectation(const uint64_t *px, const uint64_t *pz) {
        for (int p = n; p < 2 * n; ++p)
            if (symplectic_with(p, px, pz)) return 0;
        const int s = 2 * n;
        clear_row(s);
        for (int h = 0; h < n; ++h)
            if (symplectic_with(h, px, pz)) rowsum(s, h + n);
        // Scratch row now equals (-1)^{r_s} P.
        return r[s] ? -1 : +1;
    }

    // ---------------------------------------------------------------
    // Draw `shots` full Z-basis samples WITHOUT changing this tableau.
    //
    // Fact: the outcome distribution of a stabilizer state is uniform over
    // the affine subspace  v0 XOR rowspace(X-block of the stabilizers),
    // where v0 is any one possible outcome.  So:
    //   1. v0  = measure every qubit on a copy, choosing 0 for every coin;
    //   2. basis = GF(2) Gaussian elimination of the n stabilizer X-rows
    //              (k independent rows remain, k = number of random bits);
    //   3. each shot = v0 XOR (basis rows selected by k random bits).
    // rand: shots*n bytes of 0/1 from Python (first k of each shot used).
    // out : shots*n bytes, out[s*n + q] = bit of qubit q in shot s.
    // Cost: O(n^3) once + O(shots * k * n/64).
    // ---------------------------------------------------------------
    void sample(int shots, const uint8_t *rand, uint8_t *out) const {
        Tableau work(*this);                         // copy: real state untouched
        std::vector<uint64_t> v0(W, 0);
        for (int q = 0; q < n; ++q)
            if (work.measure(q, 0)) v0[q >> 6] |= 1ULL << (q & 63);

        // X-parts of the n stabilizer rows, packed, one row after another.
        std::vector<uint64_t> B(x.begin() + static_cast<size_t>(n) * W,
                                x.begin() + static_cast<size_t>(2 * n) * W);
        int k = 0;                                   // rank found so far
        for (int c = 0; c < n && k < n; ++c) {
            const int wc = c >> 6; const uint64_t m = 1ULL << (c & 63);
            int piv = -1;
            for (int i = k; i < n; ++i)
                if (B[static_cast<size_t>(i) * W + wc] & m) { piv = i; break; }
            if (piv < 0) continue;                   // no pivot in this column
            if (piv != k)
                for (int w = 0; w < W; ++w)
                    std::swap(B[static_cast<size_t>(piv) * W + w], B[static_cast<size_t>(k) * W + w]);
            for (int i = 0; i < n; ++i)              // clear column c in every other row
                if (i != k && (B[static_cast<size_t>(i) * W + wc] & m))
                    for (int w = 0; w < W; ++w)
                        B[static_cast<size_t>(i) * W + w] ^= B[static_cast<size_t>(k) * W + w];
            ++k;
        }

        std::vector<uint64_t> acc(W);
        for (int s = 0; s < shots; ++s) {
            acc = v0;
            const uint8_t *rb = rand + static_cast<size_t>(s) * n;
            for (int j = 0; j < k; ++j)
                if (rb[j] & 1)
                    for (int w = 0; w < W; ++w) acc[w] ^= B[static_cast<size_t>(j) * W + w];
            uint8_t *o = out + static_cast<size_t>(s) * n;
            for (int q = 0; q < n; ++q) o[q] = (acc[q >> 6] >> (q & 63)) & 1ULL;
        }
    }
};

inline Tableau *T(void *p) { return static_cast<Tableau *>(p); }

}  // namespace

// =====================================================================
// C ABI for ctypes
// =====================================================================
extern "C" {

void *stab_new(int n) { return new Tableau(n); }
void stab_free(void *t) { delete T(t); }
void *stab_copy(void *t) { return new Tableau(*T(t)); }

void stab_h(void *t, int a) { T(t)->h(a); }
void stab_s(void *t, int a) { T(t)->s(a); }
void stab_x(void *t, int a) { T(t)->pauli(a, 1); }
void stab_y(void *t, int a) { T(t)->pauli(a, 2); }
void stab_z(void *t, int a) { T(t)->pauli(a, 3); }
void stab_cnot(void *t, int a, int b) { T(t)->cnot(a, b); }
void stab_cz(void *t, int a, int b) { T(t)->cz(a, b); }
void stab_swap(void *t, int a, int b) { T(t)->swap(a, b); }

int stab_measure(void *t, int a, int rand_bit) { return T(t)->measure(a, rand_bit); }
int stab_is_random(void *t, int a) { return T(t)->is_random(a); }

// px, pz: n bytes each (0/1).
int stab_expectation(void *t, const uint8_t *px, const uint8_t *pz) {
    Tableau *tab = T(t);
    std::vector<uint64_t> wx(tab->W, 0), wz(tab->W, 0);
    for (int j = 0; j < tab->n; ++j) {
        if (px[j]) wx[j >> 6] |= 1ULL << (j & 63);
        if (pz[j]) wz[j >> 6] |= 1ULL << (j & 63);
    }
    return tab->expectation(wx.data(), wz.data());
}

// rand, out: shots * n bytes each (see Tableau::sample).
void stab_sample(void *t, int shots, const uint8_t *rand, uint8_t *out) {
    T(t)->sample(shots, rand, out);
}

// Export the full tableau as bytes: xs, zs are (2n) * n, rs is 2n.
void stab_dump(void *t, uint8_t *xs, uint8_t *zs, uint8_t *rs) {
    Tableau *tab = T(t);
    const int n = tab->n;
    for (int i = 0; i < 2 * n; ++i) {
        for (int j = 0; j < n; ++j) {
            xs[static_cast<size_t>(i) * n + j] = static_cast<uint8_t>(tab->get_x(i, j));
            zs[static_cast<size_t>(i) * n + j] = static_cast<uint8_t>(tab->get_z(i, j));
        }
        rs[i] = tab->r[i];
    }
}

}  