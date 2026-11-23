#include "ninfer/ops/inkling2.h"

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

std::uint16_t f32_to_bf16(float f) {
    const __nv_bfloat16 b = __float2bfloat16_rn(f);
    std::uint16_t u       = 0;
    std::memcpy(&u, &b, 2);
    return u;
}

float bf16_to_f32(std::uint16_t u) {
    __nv_bfloat16 b;
    std::memcpy(&b, &u, 2);
    return __bfloat162float(b);
}

double e4m3_to_f64(std::uint8_t code) {
    const int sign = (code & 0x80) ? -1 : 1;
    const int exp  = (code & 0x7F) >> 3;
    const int man  = code & 0x07;
    if (exp == 15 && man == 7) { return std::numeric_limits<double>::quiet_NaN(); }
    double val = (exp == 0) ? std::ldexp(static_cast<double>(man), -9)
                            : std::ldexp(1.0 + static_cast<double>(man) / 8.0, exp - 7);
    return sign * val;
}

double e2m1_to_f64(std::uint8_t nib) {
    static constexpr double kLut[16] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };
    return kLut[nib & 0xF];
}

double ue8m0_to_f64(std::uint8_t b) { return std::ldexp(1.0, static_cast<int>(b) - 127); }

struct DevBuf {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    DevBuf() = default;
    explicit DevBuf(std::size_t n) : bytes(n) {
        if (n > 0) { CUDA_CHECK(cudaMalloc(&ptr, n)); }
    }
    ~DevBuf() {
        if (ptr) { (void)cudaFree(ptr); }
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }

    template <typename T>
    void upload(const std::vector<T>& v) {
        CUDA_CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    template <typename T>
    std::vector<T> download(std::size_t count) const {
        std::vector<T> v(count);
        CUDA_CHECK(cudaMemcpy(v.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void zero() {
        if (bytes > 0) {
            // Legacy-stream memset must complete before kernels on the non-blocking test stream.
            CUDA_CHECK(cudaMemset(ptr, 0, bytes));
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
        }
    }
};

Tensor make_t(void* p, DType dt, std::int32_t ne0, std::int32_t ne1 = 1, std::int32_t ne2 = 1,
              std::int32_t ne3 = 1) {
    return Tensor(p, dt, {ne0, ne1, ne2, ne3});
}

std::vector<std::uint16_t> rand_bf16(std::mt19937& rng, std::size_t n, float scale = 0.5f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<std::uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = f32_to_bf16(dist(rng)); }
    return out;
}

std::vector<float> rand_f32(std::mt19937& rng, std::size_t n, float scale = 0.5f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = dist(rng); }
    return out;
}

void check_close(const char* name, const std::vector<double>& ref, const std::vector<double>& got,
                 double abs_tol, double rel_tol) {
    if (ref.size() != got.size()) {
        throw std::runtime_error(std::string(name) + ": size mismatch");
    }
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
            throw std::runtime_error(std::string(name) + ": non-finite output at " +
                                     std::to_string(i));
        }
        const double diff = std::abs(ref[i] - got[i]);
        const double tol  = abs_tol + rel_tol * std::abs(ref[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst_i = i;
        }
        const double rel = diff / std::max(1e-6, std::abs(ref[i]));
        if (rel > max_rel) { max_rel = rel; }
        if (diff > tol) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "%s FAILED at idx %zu: ref=%.6f got=%.6f diff=%.6f (tol=%.6f)",
                          name, i, ref[i], got[i], diff, tol);
            throw std::runtime_error(buf);
        }
    }
    std::printf("  [PASS] %-42s max_abs=%.5f (at %zu: ref=%.4f got=%.4f)\n", name, max_abs,
                worst_i, ref[worst_i], got[worst_i]);
}

std::vector<double> to_f64_from_bf16(const std::vector<std::uint16_t>& v) {
    std::vector<double> r(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { r[i] = static_cast<double>(bf16_to_f32(v[i])); }
    return r;
}

std::vector<double> to_f64_from_f32(const std::vector<float>& v) {
    std::vector<double> r(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { r[i] = static_cast<double>(v[i]); }
    return r;
}

// ============================================================================
// Test 1: BF16 Linear (T=1, 4, 8, 32, 64, 100; non-multiple-of-32 N; bias & residual)
//       + Unembed + Markov Folded Logits Argmax
// ============================================================================
void test_bf16_linear_and_heads(cudaStream_t stream) {
    std::mt19937 rng(42);
    struct Case {
        int t, n, k;
        bool bias, residual, out_f32;
    };
    const std::vector<Case> cases = {
        {1, 16, 2304, false, false, false},
        {1, 128, 2304, false, false, true},
        {1, 3276, 2304, true, false, false},
        {1, 2304, 1536, false, true, false},
        {4, 3276, 2304, false, false, false},
        {8, 128, 2304, true, false, true},
        {8, 4608, 64, false, false, false},
        {32, 3276, 2304, true, true, false},
        {64, 2304, 1536, false, true, false},
        {100, 3276, 512, true, false, false},
    };
    for (const auto& c : cases) {
        auto hx = rand_bf16(rng, static_cast<std::size_t>(c.t) * c.k, 0.25f);
        auto hw = rand_bf16(rng, static_cast<std::size_t>(c.n) * c.k, 0.25f);
        auto hb = rand_bf16(rng, c.n, 0.2f);
        auto hr = rand_bf16(rng, static_cast<std::size_t>(c.t) * c.n, 0.2f);

        DevBuf dx(hx.size() * 2), dw(hw.size() * 2), db(hb.size() * 2), dr(hr.size() * 2);
        DevBuf dout(static_cast<std::size_t>(c.t) * c.n * (c.out_f32 ? 4 : 2));
        dx.upload(hx);
        dw.upload(hw);
        db.upload(hb);
        dr.upload(hr);

        Tensor tx = make_t(dx.ptr, DType::BF16, c.k, c.t);
        Tensor tw = make_t(dw.ptr, DType::BF16, c.k, c.n);
        Tensor tb = make_t(db.ptr, DType::BF16, c.n);
        Tensor tr = make_t(dr.ptr, DType::BF16, c.n, c.t);
        Tensor to = make_t(dout.ptr, c.out_f32 ? DType::FP32 : DType::BF16, c.n, c.t);

        ops::inkling2_bf16_linear(tx, tw, to, stream, c.bias ? &tb : nullptr,
                                  c.residual ? &tr : nullptr);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> ref(static_cast<std::size_t>(c.t) * c.n, 0.0);
        for (int ti = 0; ti < c.t; ++ti) {
            for (int ni = 0; ni < c.n; ++ni) {
                double acc = 0.0;
                for (int ki = 0; ki < c.k; ++ki) {
                    acc += static_cast<double>(bf16_to_f32(hx[ti * c.k + ki])) *
                           static_cast<double>(bf16_to_f32(hw[ni * c.k + ki]));
                }
                if (c.bias) { acc += static_cast<double>(bf16_to_f32(hb[ni])); }
                if (c.residual) { acc += static_cast<double>(bf16_to_f32(hr[ti * c.n + ni])); }
                ref[ti * c.n + ni] = acc;
            }
        }
        std::vector<double> got = c.out_f32 ? to_f64_from_f32(dout.download<float>(ref.size()))
                                            : to_f64_from_bf16(dout.download<std::uint16_t>(ref.size()));
        char label[96];
        std::snprintf(label, sizeof(label), "bf16_linear T=%d N=%d K=%d %s", c.t, c.n, c.k,
                      c.out_f32 ? "F32" : "BF16");
        check_close(label, ref, got, c.out_f32 ? 2e-2 : 4e-2, c.out_f32 ? 1e-2 : 2e-2);
    }

    // Unembed + softcap + valid_cols mask + argmax (T=1 and T=4)
    for (int t : {1, 4}) {
        const int n = 4096, k = 256, valid = 4000;
        const float softcap = 30.0f;
        auto hx = rand_bf16(rng, t * k, 0.5f);
        auto hw = rand_bf16(rng, n * k, 0.5f);
        DevBuf dx(hx.size() * 2), dw(hw.size() * 2), dlog(t * n * 4), dv(t * 4), di(t * 4);
        dx.upload(hx);
        dw.upload(hw);
        Tensor tx = make_t(dx.ptr, DType::BF16, k, t);
        Tensor tw = make_t(dw.ptr, DType::BF16, k, n);
        Tensor tl = make_t(dlog.ptr, DType::FP32, n, t);
        Tensor tv = make_t(dv.ptr, DType::FP32, t);
        Tensor ti = make_t(di.ptr, DType::I32, t);
        ops::inkling2_unembed_project(tx, tw, softcap, valid, &tl, &tv, &ti, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto glog = dlog.download<float>(t * n);
        auto gval = dv.download<float>(t);
        auto gid  = di.download<std::int32_t>(t);
        for (int row = 0; row < t; ++row) {
            double best_v = -1e30;
            int best_id   = -1;
            for (int col = 0; col < n; ++col) {
                double acc = 0.0;
                for (int ki = 0; ki < k; ++ki) {
                    acc += static_cast<double>(bf16_to_f32(hx[row * k + ki])) *
                           static_cast<double>(bf16_to_f32(hw[col * k + ki]));
                }
                if (col < valid) {
                    double v = softcap * std::tanh(acc / softcap);
                    if (std::abs(v - glog[row * n + col]) > 3e-2) {
                        throw std::runtime_error("unembed logit mismatch");
                    }
                    if (v > best_v) {
                        best_v  = v;
                        best_id = col;
                    }
                } else if (!std::isinf(glog[row * n + col]) || glog[row * n + col] > 0) {
                    throw std::runtime_error("unembed invalid col not masked to -inf");
                }
            }
            if (gid[row] != best_id || std::abs(gval[row] - best_v) > 3e-2) {
                throw std::runtime_error("unembed argmax mismatch");
            }
        }
        std::printf("  [PASS] unembed_project T=%d (softcap + mask + argmax)\n", t);
    }

    // Markov folded logits argmax
    {
        const int n = 8192, valid = 8000, batch = 3;
        auto hbase = rand_f32(rng, batch * n, 2.0f);
        auto hlat  = rand_bf16(rng, batch * 256, 0.4f);
        auto hfw   = rand_bf16(rng, n * 256, 0.3f);
        auto hfb   = rand_f32(rng, n, 0.5f);
        // Make a clear unique winner per batch item inside [0, valid)
        for (int b = 0; b < batch; ++b) { hbase[b * n + (1000 + b * 777)] = 25.0f; }
        // Make an even higher value past valid to ensure mask works
        hbase[0 * n + 8050] = 999.0f;

        DevBuf dbase(hbase.size() * 4), dlat(hlat.size() * 2), dfw(hfw.size() * 2),
            dfb(hfb.size() * 4), dv(batch * 4), di(batch * 4);
        dbase.upload(hbase);
        dlat.upload(hlat);
        dfw.upload(hfw);
        dfb.upload(hfb);
        Tensor tbase = make_t(dbase.ptr, DType::FP32, n, batch);
        Tensor tlat  = make_t(dlat.ptr, DType::BF16, 256, batch);
        Tensor tfw   = make_t(dfw.ptr, DType::BF16, 256, n);
        Tensor tfb   = make_t(dfb.ptr, DType::FP32, n);
        Tensor tv    = make_t(dv.ptr, DType::FP32, batch);
        Tensor ti    = make_t(di.ptr, DType::I32, batch);
        ops::inkling2_markov_logits_argmax(tbase, tlat, tfw, tfb, valid, tv, ti, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto gv = dv.download<float>(batch);
        auto gi = di.download<std::int32_t>(batch);
        for (int b = 0; b < batch; ++b) {
            double best = -1e30;
            int best_i  = -1;
            for (int v = 0; v < valid; ++v) {
                double dot = 0.0;
                for (int r = 0; r < 256; ++r) {
                    dot += static_cast<double>(bf16_to_f32(hlat[b * 256 + r])) *
                           static_cast<double>(bf16_to_f32(hfw[v * 256 + r]));
                }
                double tot = hbase[b * n + v] + dot + hfb[v];
                if (tot > best) {
                    best   = tot;
                    best_i = v;
                }
            }
            if (gi[b] != best_i || std::abs(gv[b] - best) > 1e-2) {
                throw std::runtime_error("markov_logits_argmax mismatch");
            }
        }
        std::printf("  [PASS] markov_logits_argmax (B=3, N=8192, masked)\n");
    }

    // Paired & Triple BF16 Linear (Wave 2 fused kernels across T=1, 4, 8, 16)
    for (int t : {1, 4, 8, 16}) {
        const int k = 256, n0 = 160, n1 = 32, n2 = 48;
        auto hx0 = rand_bf16(rng, t * k, 0.25f);
        auto hx1 = rand_bf16(rng, t * k, 0.25f);
        auto hx2 = rand_bf16(rng, t * k, 0.25f);
        auto hw0 = rand_bf16(rng, n0 * k, 0.25f);
        auto hw1 = rand_bf16(rng, n1 * k, 0.25f);
        auto hw2 = rand_bf16(rng, n2 * k, 0.25f);
        DevBuf dx0(hx0.size() * 2), dx1(hx1.size() * 2), dx2(hx2.size() * 2);
        DevBuf dw0(hw0.size() * 2), dw1(hw1.size() * 2), dw2(hw2.size() * 2);
        DevBuf do0(t * n0 * 2), do1(t * n1 * 2), do2(t * n2 * 2);
        dx0.upload(hx0); dx1.upload(hx1); dx2.upload(hx2);
        dw0.upload(hw0); dw1.upload(hw1); dw2.upload(hw2);
        Tensor tx0 = make_t(dx0.ptr, DType::BF16, k, t), tw0 = make_t(dw0.ptr, DType::BF16, k, n0), to0 = make_t(do0.ptr, DType::BF16, n0, t);
        Tensor tx1 = make_t(dx1.ptr, DType::BF16, k, t), tw1 = make_t(dw1.ptr, DType::BF16, k, n1), to1 = make_t(do1.ptr, DType::BF16, n1, t);
        Tensor tx2 = make_t(dx2.ptr, DType::BF16, k, t), tw2 = make_t(dw2.ptr, DType::BF16, k, n2), to2 = make_t(do2.ptr, DType::BF16, n2, t);

        ops::inkling2_bf16_linear_triple(tx0, tw0, to0, tx1, tw1, to1, tx2, tw2, to2, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto check_mat = [&](const char* tag, const std::vector<std::uint16_t>& hx,
                             const std::vector<std::uint16_t>& hw, int n, const DevBuf& dout) {
            std::vector<double> ref(t * n, 0.0);
            for (int ti = 0; ti < t; ++ti)
                for (int ni = 0; ni < n; ++ni)
                    for (int ki = 0; ki < k; ++ki)
                        ref[ti * n + ni] += static_cast<double>(bf16_to_f32(hx[ti * k + ki])) *
                                            static_cast<double>(bf16_to_f32(hw[ni * k + ki]));
            check_close(tag, ref, to_f64_from_bf16(dout.download<std::uint16_t>(t * n)), 3e-2, 2e-2);
        };
        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "bf16_linear_triple T=%d (seg0)", t);
        check_mat(lbl, hx0, hw0, n0, do0);
        std::snprintf(lbl, sizeof(lbl), "bf16_linear_triple T=%d (seg1)", t);
        check_mat(lbl, hx1, hw1, n1, do1);
        std::snprintf(lbl, sizeof(lbl), "bf16_linear_triple T=%d (seg2)", t);
        check_mat(lbl, hx2, hw2, n2, do2);
    }
    // Pair split-K kernel (K=512, N=16)
    for (int t : {1, 4, 8}) {
        const int k = 512, n0 = 16, n1 = 16;
        auto hx0 = rand_bf16(rng, t * k, 0.25f), hx1 = rand_bf16(rng, t * k, 0.25f);
        auto hw0 = rand_bf16(rng, n0 * k, 0.25f), hw1 = rand_bf16(rng, n1 * k, 0.25f);
        DevBuf dx0(hx0.size() * 2), dx1(hx1.size() * 2), dw0(hw0.size() * 2), dw1(hw1.size() * 2),
            do0(t * n0 * 2), do1(t * n1 * 2);
        dx0.upload(hx0); dx1.upload(hx1); dw0.upload(hw0); dw1.upload(hw1);
        Tensor tx0 = make_t(dx0.ptr, DType::BF16, k, t), tw0 = make_t(dw0.ptr, DType::BF16, k, n0), to0 = make_t(do0.ptr, DType::BF16, n0, t);
        Tensor tx1 = make_t(dx1.ptr, DType::BF16, k, t), tw1 = make_t(dw1.ptr, DType::BF16, k, n1), to1 = make_t(do1.ptr, DType::BF16, n1, t);
        ops::inkling2_bf16_linear_pair(tx0, tw0, to0, tx1, tw1, to1, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> ref0(t * n0, 0.0), ref1(t * n1, 0.0);
        for (int ti = 0; ti < t; ++ti)
            for (int ni = 0; ni < n0; ++ni)
                for (int ki = 0; ki < k; ++ki) {
                    ref0[ti * n0 + ni] += static_cast<double>(bf16_to_f32(hx0[ti * k + ki])) * bf16_to_f32(hw0[ni * k + ki]);
                    ref1[ti * n1 + ni] += static_cast<double>(bf16_to_f32(hx1[ti * k + ki])) * bf16_to_f32(hw1[ni * k + ki]);
                }
        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "bf16_linear_pair T=%d", t);
        check_close(lbl, ref0, to_f64_from_bf16(do0.download<std::uint16_t>(t * n0)), 3e-2, 2e-2);
        check_close(lbl, ref1, to_f64_from_bf16(do1.download<std::uint16_t>(t * n1)), 3e-2, 2e-2);
    }

    // 5-way BF16 Linear (Wave 4.5 fused global-layer projection across T=1, 4, 8, 16, 20)
    for (int t : {1, 4, 8, 16, 20}) {
        const int k = 256;
        const int ns[5] = {160, 32, 48, 16, 16};
        std::vector<std::vector<std::uint16_t>> hx(5), hw(5);
        std::vector<DevBuf> dx, dw, dout;
        for (int s = 0; s < 5; ++s) {
            hx[s] = rand_bf16(rng, t * k, 0.25f);
            hw[s] = rand_bf16(rng, ns[s] * k, 0.25f);
            dx.emplace_back(hx[s].size() * 2); dx.back().upload(hx[s]);
            dw.emplace_back(hw[s].size() * 2); dw.back().upload(hw[s]);
            dout.emplace_back(t * ns[s] * 2);
        }
        Tensor tx0 = make_t(dx[0].ptr, DType::BF16, k, t), tw0 = make_t(dw[0].ptr, DType::BF16, k, ns[0]), to0 = make_t(dout[0].ptr, DType::BF16, ns[0], t);
        Tensor tx1 = make_t(dx[1].ptr, DType::BF16, k, t), tw1 = make_t(dw[1].ptr, DType::BF16, k, ns[1]), to1 = make_t(dout[1].ptr, DType::BF16, ns[1], t);
        Tensor tx2 = make_t(dx[2].ptr, DType::BF16, k, t), tw2 = make_t(dw[2].ptr, DType::BF16, k, ns[2]), to2 = make_t(dout[2].ptr, DType::BF16, ns[2], t);
        Tensor tx3 = make_t(dx[3].ptr, DType::BF16, k, t), tw3 = make_t(dw[3].ptr, DType::BF16, k, ns[3]), to3 = make_t(dout[3].ptr, DType::BF16, ns[3], t);
        Tensor tx4 = make_t(dx[4].ptr, DType::BF16, k, t), tw4 = make_t(dw[4].ptr, DType::BF16, k, ns[4]), to4 = make_t(dout[4].ptr, DType::BF16, ns[4], t);
        ops::inkling2_bf16_linear_5way(tx0, tw0, to0, tx1, tw1, to1, tx2, tw2, to2, tx3, tw3, to3,
                                       tx4, tw4, to4, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int s = 0; s < 5; ++s) {
            std::vector<double> ref(t * ns[s], 0.0);
            for (int ti = 0; ti < t; ++ti)
                for (int ni = 0; ni < ns[s]; ++ni)
                    for (int ki = 0; ki < k; ++ki)
                        ref[ti * ns[s] + ni] += static_cast<double>(bf16_to_f32(hx[s][ti * k + ki])) *
                                                static_cast<double>(bf16_to_f32(hw[s][ni * k + ki]));
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "bf16_linear_5way T=%d seg=%d", t, s);
            check_close(lbl, ref, to_f64_from_bf16(dout[s].download<std::uint16_t>(t * ns[s])), 3e-2, 2e-2);
        }
    }

    // Wave 4 fused argmax-only unembed_project (logits_out == nullptr, T=1, 4, 8)
    for (int t : {1, 4, 8}) {
        const int n = 4096, k = 256, valid = 4000;
        const float softcap = 30.0f;
        auto hx = rand_bf16(rng, t * k, 0.5f);
        auto hw = rand_bf16(rng, n * k, 0.5f);
        DevBuf dx(hx.size() * 2), dw(hw.size() * 2), dv(t * 4), di(t * 4);
        dx.upload(hx); dw.upload(hw);
        Tensor tx = make_t(dx.ptr, DType::BF16, k, t);
        Tensor tw = make_t(dw.ptr, DType::BF16, k, n);
        Tensor tv = make_t(dv.ptr, DType::FP32, t);
        Tensor ti = make_t(di.ptr, DType::I32, t);
        ops::inkling2_unembed_project(tx, tw, softcap, valid, nullptr, &tv, &ti, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto gval = dv.download<float>(t);
        auto gid  = di.download<std::int32_t>(t);
        for (int row = 0; row < t; ++row) {
            double best_v = -1e30;
            int best_id   = -1;
            for (int col = 0; col < valid; ++col) {
                double acc = 0.0;
                for (int ki = 0; ki < k; ++ki) {
                    acc += static_cast<double>(bf16_to_f32(hx[row * k + ki])) *
                           static_cast<double>(bf16_to_f32(hw[col * k + ki]));
                }
                double v = softcap * std::tanh(acc / softcap);
                if (v > best_v) { best_v = v; best_id = col; }
            }
            if (gid[row] != best_id || std::abs(gval[row] - best_v) > 3e-2) {
                throw std::runtime_error("unembed argmax-only mismatch");
            }
        }
        std::printf("  [PASS] unembed_project argmax-only T=%d (pf2)\n", t);
    }

    // Wave 5: direct BF16 unembed_project (T=1, 4, 8, 12) bit-exact vs FP32 softcap_mask + round-to-BF16
    for (int t : {1, 4, 8, 12}) {
        const int n = 4096, k = 256, valid = 4000;
        const float softcap = 30.0f;
        auto hx = rand_bf16(rng, t * k, 0.5f);
        auto hw = rand_bf16(rng, n * k, 0.5f);
        DevBuf dx(hx.size() * 2), dw(hw.size() * 2), df32(t * n * 4), dbf16(t * n * 2);
        dx.upload(hx); dw.upload(hw);
        Tensor tx    = make_t(dx.ptr, DType::BF16, k, t);
        Tensor tw    = make_t(dw.ptr, DType::BF16, k, n);
        Tensor t_f32 = make_t(df32.ptr, DType::FP32, n, t);
        Tensor t_b16 = make_t(dbf16.ptr, DType::BF16, n, t);

        // Exercises multi-CTA softcap_mask_f32_kernel (no argmax) and fused direct-BF16 unembed!
        ops::inkling2_unembed_project(tx, tw, softcap, valid, &t_f32, nullptr, nullptr, stream);
        ops::inkling2_unembed_project(tx, tw, softcap, valid, &t_b16, nullptr, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto gf32 = df32.download<float>(t * n);
        auto gb16 = dbf16.download<std::uint16_t>(t * n);
        int bit_diffs = 0;
        for (int row = 0; row < t; ++row) {
            for (int col = 0; col < n; ++col) {
                const std::size_t idx = static_cast<std::size_t>(row) * n + col;
                if (col >= valid) {
                    if (!std::isinf(gf32[idx]) || gf32[idx] > 0.0f || gb16[idx] != 0xFF80u) {
                        throw std::runtime_error("unembed BF16/FP32 invalid col not masked to -inf");
                    }
                } else {
                    const std::uint16_t want = f32_to_bf16(gf32[idx]);
                    if (t <= 8) {
                        if (gb16[idx] != want) {
                            if (bit_diffs < 5) {
                                std::fprintf(stderr,
                                             "  [DIFF] T=%d row=%d col=%d f32=%.9g want=0x%04x got=0x%04x\n",
                                             t, row, col, gf32[idx], want, gb16[idx]);
                            }
                            ++bit_diffs;
                        }
                    } else if (std::abs(bf16_to_f32(gb16[idx]) - gf32[idx]) > 4e-2f) {
                        throw std::runtime_error("unembed BF16 T>8 value mismatch");
                    }
                }
            }
        }
        if (bit_diffs != 0) {
            throw std::runtime_error("unembed direct BF16 bit_diffs != 0 vs FP32+round");
        }
        std::printf("  [PASS] unembed_project direct BF16 T=%d (bit-exact vs FP32+round)\n", t);
    }
}

// ============================================================================
// Test 2: RMSNorm, Sharded Embed, Block AttnRes, Loop Controller
// ============================================================================
void test_norms_attnres_controller(cudaStream_t stream) {
    std::mt19937 rng(101);
    const int H = 2304, T = 5;
    const float eps = 1e-6f;

    // Plain RMSNorm
    {
        auto hx = rand_bf16(rng, T * H, 1.0f);
        auto hw = rand_bf16(rng, H, 1.0f);
        DevBuf dx(hx.size() * 2), dw(hw.size() * 2), dout(hx.size() * 2);
        dx.upload(hx);
        dw.upload(hw);
        Tensor tx = make_t(dx.ptr, DType::BF16, H, T);
        Tensor tw = make_t(dw.ptr, DType::BF16, H);
        Tensor to = make_t(dout.ptr, DType::BF16, H, T);
        ops::inkling2_rmsnorm(tx, &tw, to, eps, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> ref(T * H);
        for (int t = 0; t < T; ++t) {
            double sq = 0.0;
            for (int i = 0; i < H; ++i) {
                double v = bf16_to_f32(hx[t * H + i]);
                sq += v * v;
            }
            double inv = 1.0 / std::sqrt(sq / H + eps);
            for (int i = 0; i < H; ++i) {
                ref[t * H + i] = bf16_to_f32(hx[t * H + i]) * inv * bf16_to_f32(hw[i]);
            }
        }
        check_close("rmsnorm (no +1 offset)", ref, to_f64_from_bf16(dout.download<std::uint16_t>(T * H)),
                    2e-2, 1.5e-2);
    }

    // Block AttnRes (S=4, Q=3 with fused post-RMSNorm)
    {
        const int S = 4, Q = 3;
        std::vector<std::vector<std::uint16_t>> hsrc(S), hnw(Q);
        std::vector<DevBuf> dsrc, dnw, dout;
        std::vector<const void*> src_ptrs(S), nw_ptrs(Q);
        std::vector<void*> out_ptrs(Q);
        for (int s = 0; s < S; ++s) {
            hsrc[s] = rand_bf16(rng, T * H, 1.0f);
            dsrc.emplace_back(T * H * 2);
            dsrc.back().upload(hsrc[s]);
            src_ptrs[s] = dsrc.back().ptr;
        }
        for (int q = 0; q < Q; ++q) {
            hnw[q] = rand_bf16(rng, H, 1.0f);
            dnw.emplace_back(H * 2);
            dnw.back().upload(hnw[q]);
            nw_ptrs[q] = dnw.back().ptr;
            dout.emplace_back(T * H * 2);
            out_ptrs[q] = dout.back().ptr;
        }
        auto hq = rand_f32(rng, Q * H, 1.0f);
        DevBuf dq(hq.size() * 4);
        dq.upload(hq);
        Tensor tq = make_t(dq.ptr, DType::FP32, H, Q);

        ops::inkling2_attn_res_mix(src_ptrs.data(), S, tq, nw_ptrs.data(), out_ptrs.data(), Q, H,
                                   T, eps, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int q = 0; q < Q; ++q) {
            std::vector<double> ref(T * H);
            for (int t = 0; t < T; ++t) {
                std::vector<std::vector<double>> v(S, std::vector<double>(H));
                for (int s = 0; s < S; ++s) {
                    double sq = 0.0;
                    for (int i = 0; i < H; ++i) {
                        double x = bf16_to_f32(hsrc[s][t * H + i]);
                        v[s][i]  = x;
                        sq += x * x;
                    }
                    double inv = 1.0 / std::sqrt(sq / H + eps);
                    for (int i = 0; i < H; ++i) { v[s][i] *= inv; }
                }
                std::vector<double> scores(S), probs(S);
                double mx = -1e30;
                for (int s = 0; s < S; ++s) {
                    double dot = 0.0;
                    for (int i = 0; i < H; ++i) { dot += hq[q * H + i] * v[s][i]; }
                    scores[s] = dot / std::sqrt(static_cast<double>(H));
                    mx        = std::max(mx, scores[s]);
                }
                double den = 0.0;
                for (int s = 0; s < S; ++s) {
                    probs[s] = std::exp(scores[s] - mx);
                    den += probs[s];
                }
                for (int s = 0; s < S; ++s) { probs[s] /= den; }
                double sq = 0.0;
                std::vector<double> mixed(H, 0.0);
                for (int i = 0; i < H; ++i) {
                    for (int s = 0; s < S; ++s) { mixed[i] += probs[s] * v[s][i]; }
                    sq += mixed[i] * mixed[i];
                }
                double inv = 1.0 / std::sqrt(sq / H + eps);
                for (int i = 0; i < H; ++i) {
                    ref[t * H + i] = mixed[i] * inv * bf16_to_f32(hnw[q][i]);
                }
            }
            check_close("attn_res_mix (S=4, Q=3, fused norm)", ref,
                        to_f64_from_bf16(dout[q].download<std::uint16_t>(T * H)), 2e-2, 2e-2);
        }
    }

    // Cross-pass Loop Controller (with fused post-RMSNorm)
    {
        auto hcur = rand_bf16(rng, T * H, 1.0f);
        auto hmsg = rand_bf16(rng, T * H, 1.0f);
        auto hdn  = rand_bf16(rng, 64 * 2 * H, 0.1f);
        auto hgf  = rand_bf16(rng, 2 * H * 64, 0.1f);
        auto hnw  = rand_bf16(rng, H, 1.0f);

        DevBuf dcur(hcur.size() * 2), dmsg(hmsg.size() * 2), ddn(hdn.size() * 2),
            dgf(hgf.size() * 2), dnw(hnw.size() * 2), dz(T * 2 * H * 2), dh(T * 64 * 2),
            dgfo(T * 2 * H * 2), dout(T * H * 2);
        dcur.upload(hcur);
        dmsg.upload(hmsg);
        ddn.upload(hdn);
        dgf.upload(hgf);
        dnw.upload(hnw);

        Tensor tcur = make_t(dcur.ptr, DType::BF16, H, T);
        Tensor tmsg = make_t(dmsg.ptr, DType::BF16, H, T);
        Tensor tdn  = make_t(ddn.ptr, DType::BF16, 2 * H, 64);
        Tensor tgf  = make_t(dgf.ptr, DType::BF16, 64, 2 * H);
        Tensor tnw  = make_t(dnw.ptr, DType::BF16, H);
        Tensor tz   = make_t(dz.ptr, DType::BF16, 2 * H, T);
        Tensor th   = make_t(dh.ptr, DType::BF16, 64, T);
        Tensor tgfo = make_t(dgfo.ptr, DType::BF16, 2 * H, T);
        Tensor to   = make_t(dout.ptr, DType::BF16, H, T);

        ops::inkling2_loop_control(tcur, tmsg, tdn, tgf, &tnw, tz, th, tgfo, to, eps, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> ref(T * H);
        for (int t = 0; t < T; ++t) {
            std::vector<double> z(2 * H);
            double sqc = 0, sqm = 0;
            for (int i = 0; i < H; ++i) {
                double c = bf16_to_f32(hcur[t * H + i]);
                double m = bf16_to_f32(hmsg[t * H + i]);
                sqc += c * c;
                sqm += m * m;
            }
            double ic = 1.0 / std::sqrt(sqc / H + eps);
            double im = 1.0 / std::sqrt(sqm / H + eps);
            for (int i = 0; i < H; ++i) {
                z[i]     = bf16_to_f32(hcur[t * H + i]) * ic;
                z[H + i] = bf16_to_f32(hmsg[t * H + i]) * im;
            }
            std::vector<double> h(64, 0.0);
            for (int r = 0; r < 64; ++r) {
                for (int i = 0; i < 2 * H; ++i) { h[r] += z[i] * bf16_to_f32(hdn[r * 2 * H + i]); }
            }
            std::vector<double> out_v(H);
            double sq = 0.0;
            for (int i = 0; i < H; ++i) {
                double gv = 0.0, fv = 0.0;
                for (int r = 0; r < 64; ++r) {
                    gv += h[r] * bf16_to_f32(hgf[i * 64 + r]);
                    fv += h[r] * bf16_to_f32(hgf[(H + i) * 64 + r]);
                }
                out_v[i] = (1.0 - std::tanh(fv)) * bf16_to_f32(hcur[t * H + i]) +
                           std::tanh(gv) * bf16_to_f32(hmsg[t * H + i]);
                sq += out_v[i] * out_v[i];
            }
            double io = 1.0 / std::sqrt(sq / H + eps);
            for (int i = 0; i < H; ++i) { ref[t * H + i] = out_v[i] * io * bf16_to_f32(hnw[i]); }
        }
        check_close("loop_control (fused post-RMSNorm)", ref,
                    to_f64_from_bf16(dout.download<std::uint16_t>(T * H)), 5e-2, 3e-2);
    }

    // Batched 3-controller loop_control_qkv (matches 3 separate loop_control calls!)
    for (int t_case : {1, 4, 8, 12}) {
        std::vector<std::vector<std::uint16_t>> hcur(3), hmsg(3), hdn(3), hgf(3), hnw(3);
        std::vector<DevBuf> dcur, dmsg, ddn, dgf, dnw, do_ref, do_qkv;
        for (int c = 0; c < 3; ++c) {
            hcur[c] = rand_bf16(rng, t_case * H, 1.0f);
            hmsg[c] = rand_bf16(rng, t_case * H, 1.0f);
            hdn[c]  = rand_bf16(rng, 64 * 2 * H, 0.1f);
            hgf[c]  = rand_bf16(rng, 2 * H * 64, 0.1f);
            hnw[c]  = rand_bf16(rng, H, 1.0f);
            dcur.emplace_back(hcur[c].size() * 2); dcur.back().upload(hcur[c]);
            dmsg.emplace_back(hmsg[c].size() * 2); dmsg.back().upload(hmsg[c]);
            ddn.emplace_back(hdn[c].size() * 2);   ddn.back().upload(hdn[c]);
            dgf.emplace_back(hgf[c].size() * 2);   dgf.back().upload(hgf[c]);
            dnw.emplace_back(hnw[c].size() * 2);   dnw.back().upload(hnw[c]);
            do_ref.emplace_back(t_case * H * 2);
            do_qkv.emplace_back(t_case * H * 2);
        }
        DevBuf dz(3 * t_case * 2 * H * 2), dh(3 * t_case * 64 * 2), dgf_sc(3 * t_case * 2 * H * 2);
        Tensor tz1 = make_t(dz.ptr, DType::BF16, 2 * H, t_case);
        Tensor th1 = make_t(dh.ptr, DType::BF16, 64, t_case);
        Tensor tg1 = make_t(dgf_sc.ptr, DType::BF16, 2 * H, t_case);
        for (int c = 0; c < 3; ++c) {
            Tensor tc = make_t(dcur[c].ptr, DType::BF16, H, t_case);
            Tensor tm = make_t(dmsg[c].ptr, DType::BF16, H, t_case);
            Tensor td = make_t(ddn[c].ptr, DType::BF16, 2 * H, 64);
            Tensor tg = make_t(dgf[c].ptr, DType::BF16, 64, 2 * H);
            Tensor tn = make_t(dnw[c].ptr, DType::BF16, H);
            Tensor to = make_t(do_ref[c].ptr, DType::BF16, H, t_case);
            ops::inkling2_loop_control(tc, tm, td, tg, &tn, tz1, th1, tg1, to, eps, stream);
        }
        Tensor tz3 = make_t(dz.ptr, DType::BF16, 2 * H, 3 * t_case);
        Tensor th3 = make_t(dh.ptr, DType::BF16, 64, 3 * t_case);
        Tensor tg3 = make_t(dgf_sc.ptr, DType::BF16, 2 * H, 3 * t_case);
        Tensor tc0 = make_t(dcur[0].ptr, DType::BF16, H, t_case), tc1 = make_t(dcur[1].ptr, DType::BF16, H, t_case), tc2 = make_t(dcur[2].ptr, DType::BF16, H, t_case);
        Tensor tm0 = make_t(dmsg[0].ptr, DType::BF16, H, t_case), tm1 = make_t(dmsg[1].ptr, DType::BF16, H, t_case), tm2 = make_t(dmsg[2].ptr, DType::BF16, H, t_case);
        Tensor td0 = make_t(ddn[0].ptr, DType::BF16, 2 * H, 64),  td1 = make_t(ddn[1].ptr, DType::BF16, 2 * H, 64),  td2 = make_t(ddn[2].ptr, DType::BF16, 2 * H, 64);
        Tensor tf0 = make_t(dgf[0].ptr, DType::BF16, 64, 2 * H),  tf1 = make_t(dgf[1].ptr, DType::BF16, 64, 2 * H),  tf2 = make_t(dgf[2].ptr, DType::BF16, 64, 2 * H);
        Tensor tn0 = make_t(dnw[0].ptr, DType::BF16, H),          tn1 = make_t(dnw[1].ptr, DType::BF16, H),          tn2 = make_t(dnw[2].ptr, DType::BF16, H);
        Tensor to0 = make_t(do_qkv[0].ptr, DType::BF16, H, t_case), to1 = make_t(do_qkv[1].ptr, DType::BF16, H, t_case), to2 = make_t(do_qkv[2].ptr, DType::BF16, H, t_case);
        ops::inkling2_loop_control_qkv(tc0, tc1, tc2, tm0, tm1, tm2, td0, td1, td2, tf0, tf1, tf2,
                                       &tn0, &tn1, &tn2, tz3, th3, tg3, to0, to1, to2, eps, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int c = 0; c < 3; ++c) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "loop_control_qkv T=%d c=%d vs single", t_case, c);
            check_close(lbl, to_f64_from_bf16(do_ref[c].download<std::uint16_t>(t_case * H)),
                        to_f64_from_bf16(do_qkv[c].download<std::uint16_t>(t_case * H)), 1e-2, 1e-2);
        }
    }

    // Wave 3/4: attn_res_mix_and_prep + loop_control(skip_prep=true),
    // attn_res_mix_pair_and_prep_qkv + loop_control_qkv(skip_prep=true), and swap_pass_banks
    for (int t_case : {1, 4}) {
        const int S = 4;
        std::vector<std::vector<std::uint16_t>> hsrc(S);
        std::vector<DevBuf> dsrc;
        std::vector<const void*> sptrs(S);
        for (int s = 0; s < S; ++s) {
            hsrc[s] = rand_bf16(rng, t_case * H, 0.8f);
            dsrc.emplace_back(hsrc[s].size() * 2); dsrc.back().upload(hsrc[s]);
            sptrs[s] = dsrc.back().ptr;
        }
        auto hq   = rand_f32(rng, H, 0.7f);
        auto hmsg = rand_bf16(rng, t_case * H, 0.8f);
        auto hdn  = rand_bf16(rng, 64 * 2 * H, 0.1f);
        auto hgf  = rand_bf16(rng, 2 * H * 64, 0.1f);
        auto hnw  = rand_bf16(rng, H, 1.0f);
        DevBuf dq(H * 4), dmsg(hmsg.size() * 2), ddn(hdn.size() * 2), dgf(hgf.size() * 2), dnw(H * 2);
        DevBuf dmix_r(t_case * H * 2), dmix_f(t_case * H * 2);
        DevBuf dz_r(t_case * 2 * H * 2), dz_f(t_case * 2 * H * 2), dh(t_case * 64 * 2), dgf_s(t_case * 2 * H * 2);
        DevBuf do_r(t_case * H * 2), do_f(t_case * H * 2);
        dq.upload(hq); dmsg.upload(hmsg); ddn.upload(hdn); dgf.upload(hgf); dnw.upload(hnw);
        Tensor tq = make_t(dq.ptr, DType::FP32, H, 1), tmsg = make_t(dmsg.ptr, DType::BF16, H, t_case);
        Tensor tdn = make_t(ddn.ptr, DType::BF16, 2 * H, 64), tgf = make_t(dgf.ptr, DType::BF16, 64, 2 * H);
        Tensor tnw = make_t(dnw.ptr, DType::BF16, H);
        Tensor tmr = make_t(dmix_r.ptr, DType::BF16, H, t_case), tmf = make_t(dmix_f.ptr, DType::BF16, H, t_case);
        Tensor tzr = make_t(dz_r.ptr, DType::BF16, 2 * H, t_case), tzf = make_t(dz_f.ptr, DType::BF16, 2 * H, t_case);
        Tensor th = make_t(dh.ptr, DType::BF16, 64, t_case), tgfs = make_t(dgf_s.ptr, DType::BF16, 2 * H, t_case);
        Tensor tor = make_t(do_r.ptr, DType::BF16, H, t_case), tof = make_t(do_f.ptr, DType::BF16, H, t_case);

        const void* null_nw[1] = {nullptr};
        void* out_r[1]         = {dmix_r.ptr};
        ops::inkling2_attn_res_mix(sptrs.data(), S, tq, null_nw, out_r, 1, H, t_case, eps, stream);
        ops::inkling2_loop_control(tmr, tmsg, tdn, tgf, &tnw, tzr, th, tgfs, tor, eps, stream, false);

        ops::inkling2_attn_res_mix_and_prep(sptrs.data(), S, tq, tmsg, tmf, tzf, eps, stream);
        ops::inkling2_loop_control(tmf, tmsg, tdn, tgf, &tnw, tzf, th, tgfs, tof, eps, stream, true);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "attn_res_mix_and_prep + loop_ctrl(skip_prep) T=%d", t_case);
        check_close(lbl, to_f64_from_bf16(dmix_r.download<std::uint16_t>(t_case * H)),
                    to_f64_from_bf16(dmix_f.download<std::uint16_t>(t_case * H)), 1e-2, 1e-2);
        check_close(lbl, to_f64_from_bf16(do_r.download<std::uint16_t>(t_case * H)),
                    to_f64_from_bf16(do_f.download<std::uint16_t>(t_case * H)), 1.5e-2, 1.5e-2);

        // Paired Cross-Pass AttnRes + QKV Prep (both num_cur_q=1 and num_cur_q=3)
        for (int n_cur_q : {1, 3}) {
            const int S0 = 2, Q0 = 4, S1 = 3;
            auto hq0 = rand_f32(rng, Q0 * H, 0.7f), hq1 = rand_f32(rng, n_cur_q * H, 0.7f);
            DevBuf dq0(hq0.size() * 4), dq1(hq1.size() * 4);
            dq0.upload(hq0); dq1.upload(hq1);
            Tensor tq0 = make_t(dq0.ptr, DType::FP32, H, Q0), tq1 = make_t(dq1.ptr, DType::FP32, H, n_cur_q);

            std::vector<DevBuf> dmsg_r, dmsg_f, dcur_r, dcur_f, ddn3, dgf3, dnw3, dor3, dof3;
            void* mp_r[4]; void* mp_f[4]; void* cp_r[3]; void* cp_f[3];
            const void* nn1[3] = {nullptr, nullptr, nullptr};
            for (int i = 0; i < 4; ++i) {
                dmsg_r.emplace_back(t_case * H * 2); mp_r[i] = dmsg_r.back().ptr;
                dmsg_f.emplace_back(t_case * H * 2); mp_f[i] = dmsg_f.back().ptr;
            }
            for (int i = 0; i < 3; ++i) {
                dcur_r.emplace_back(t_case * H * 2); cp_r[i] = dcur_r.back().ptr;
                dcur_f.emplace_back(t_case * H * 2); cp_f[i] = dcur_f.back().ptr;
                auto d_w = rand_bf16(rng, 64 * 2 * H, 0.1f), g_w = rand_bf16(rng, 2 * H * 64, 0.1f), n_w = rand_bf16(rng, H, 1.0f);
                ddn3.emplace_back(d_w.size() * 2); ddn3.back().upload(d_w);
                dgf3.emplace_back(g_w.size() * 2); dgf3.back().upload(g_w);
                dnw3.emplace_back(n_w.size() * 2); dnw3.back().upload(n_w);
                dor3.emplace_back(t_case * H * 2);
                dof3.emplace_back(t_case * H * 2);
            }
            DevBuf dz3r(3 * t_case * 2 * H * 2), dz3f(3 * t_case * 2 * H * 2);
            DevBuf dh3(3 * t_case * 64 * 2), dgf3s(3 * t_case * 2 * H * 2);
            Tensor tz3r = make_t(dz3r.ptr, DType::BF16, 2 * H, 3 * t_case);
            Tensor tz3f = make_t(dz3f.ptr, DType::BF16, 2 * H, 3 * t_case);
            Tensor th3  = make_t(dh3.ptr, DType::BF16, 64, 3 * t_case);
            Tensor tgf3s = make_t(dgf3s.ptr, DType::BF16, 2 * H, 3 * t_case);

            const void* bank_srcs[2] = {sptrs[0], sptrs[1]};
            const void* cur_srcs[3]  = {sptrs[0], sptrs[1], sptrs[2]};
            ops::inkling2_attn_res_mix_pair(bank_srcs, S0, tq0, mp_r, Q0, cur_srcs, S1, tq1, nn1,
                                            cp_r, n_cur_q, H, t_case, eps, stream);
            Tensor tc0r = make_t(cp_r[0], DType::BF16, H, t_case),
                   tc1r = make_t(cp_r[n_cur_q == 1 ? 0 : 1], DType::BF16, H, t_case),
                   tc2r = make_t(cp_r[n_cur_q == 1 ? 0 : 2], DType::BF16, H, t_case);
            Tensor tm0r = make_t(mp_r[0], DType::BF16, H, t_case), tm1r = make_t(mp_r[1], DType::BF16, H, t_case), tm2r = make_t(mp_r[2], DType::BF16, H, t_case);
            Tensor td0 = make_t(ddn3[0].ptr, DType::BF16, 2 * H, 64), td1 = make_t(ddn3[1].ptr, DType::BF16, 2 * H, 64), td2 = make_t(ddn3[2].ptr, DType::BF16, 2 * H, 64);
            Tensor tf0 = make_t(dgf3[0].ptr, DType::BF16, 64, 2 * H), tf1 = make_t(dgf3[1].ptr, DType::BF16, 64, 2 * H), tf2 = make_t(dgf3[2].ptr, DType::BF16, 64, 2 * H);
            Tensor tn0 = make_t(dnw3[0].ptr, DType::BF16, H), tn1 = make_t(dnw3[1].ptr, DType::BF16, H), tn2 = make_t(dnw3[2].ptr, DType::BF16, H);
            Tensor to0r = make_t(dor3[0].ptr, DType::BF16, H, t_case), to1r = make_t(dor3[1].ptr, DType::BF16, H, t_case), to2r = make_t(dor3[2].ptr, DType::BF16, H, t_case);
            ops::inkling2_loop_control_qkv(tc0r, tc1r, tc2r, tm0r, tm1r, tm2r, td0, td1, td2,
                                           tf0, tf1, tf2, &tn0, &tn1, &tn2, tz3r, th3, tgf3s,
                                           to0r, to1r, to2r, eps, stream, false);

            ops::inkling2_attn_res_mix_pair_and_prep_qkv(bank_srcs, tq0, mp_f, cur_srcs, S1, tq1,
                                                         cp_f, n_cur_q, tz3f, eps, stream);
            Tensor tc0f = make_t(cp_f[0], DType::BF16, H, t_case),
                   tc1f = make_t(cp_f[n_cur_q == 1 ? 0 : 1], DType::BF16, H, t_case),
                   tc2f = make_t(cp_f[n_cur_q == 1 ? 0 : 2], DType::BF16, H, t_case);
            Tensor tm0f = make_t(mp_f[0], DType::BF16, H, t_case), tm1f = make_t(mp_f[1], DType::BF16, H, t_case), tm2f = make_t(mp_f[2], DType::BF16, H, t_case);
            Tensor to0f = make_t(dof3[0].ptr, DType::BF16, H, t_case), to1f = make_t(dof3[1].ptr, DType::BF16, H, t_case), to2f = make_t(dof3[2].ptr, DType::BF16, H, t_case);
            ops::inkling2_loop_control_qkv(tc0f, tc1f, tc2f, tm0f, tm1f, tm2f, td0, td1, td2,
                                           tf0, tf1, tf2, &tn0, &tn1, &tn2, tz3f, th3, tgf3s,
                                           to0f, to1f, to2f, eps, stream, true);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int c = 0; c < 3; ++c) {
                std::snprintf(lbl, sizeof(lbl), "attn_res_pair_prep_qkv T=%d Q1=%d c=%d", t_case, n_cur_q, c);
                check_close(lbl, to_f64_from_bf16(dor3[c].download<std::uint16_t>(t_case * H)),
                            to_f64_from_bf16(dof3[c].download<std::uint16_t>(t_case * H)), 1.5e-2, 1.5e-2);
            }
        }
    }

    // Wave 4: swap_pass_banks bit-exact copy
    {
        auto hsb0 = rand_bf16(rng, 4 * H, 1.0f), hpre = rand_bf16(rng, 4 * H, 1.0f), hcur = rand_bf16(rng, 4 * H, 1.0f);
        DevBuf dsb0(hsb0.size() * 2), dpre(hpre.size() * 2), dcur(hcur.size() * 2), db0(hcur.size() * 2), dbout(hcur.size() * 2);
        dsb0.upload(hsb0); dpre.upload(hpre); dcur.upload(hcur);
        db0.zero(); dbout.zero();
        Tensor tsb0 = make_t(dsb0.ptr, DType::BF16, H, 4), tpre = make_t(dpre.ptr, DType::BF16, H, 4);
        Tensor tcur = make_t(dcur.ptr, DType::BF16, H, 4), tb0 = make_t(db0.ptr, DType::BF16, H, 4), tbout = make_t(dbout.ptr, DType::BF16, H, 4);
        ops::inkling2_swap_pass_banks(tsb0, tpre, tcur, tb0, tbout, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (db0.download<std::uint16_t>(4 * H) != hsb0 ||
            dbout.download<std::uint16_t>(4 * H) != hcur ||
            dcur.download<std::uint16_t>(4 * H) != hpre) {
            throw std::runtime_error("swap_pass_banks mismatch");
        }
        std::printf("  [PASS] swap_pass_banks exact\n");
    }
}

// ============================================================================
// Test 3: Static & Dynamic SConv (prefill == sequential decode == verify + commit)
// ============================================================================
void test_sconv(cudaStream_t stream) {
    std::mt19937 rng(202);
    const int C = 512, Hkv = 4, T_prefill = 6, T_ver = 5;
    auto hx   = rand_bf16(rng, (T_prefill + T_ver) * C, 0.8f);
    auto hw   = rand_bf16(rng, C * 4, 0.5f);
    auto hres = rand_bf16(rng, (T_prefill + T_ver) * C, 0.5f);

    // Static sconv: compare prefill on full (T_prefill+T_ver) vs prefill(T_prefill) + verify(T_ver) + commit(3)
    {
        DevBuf dx(hx.size() * 2), dw(hw.size() * 2), dres(hres.size() * 2);
        DevBuf dout_full(hx.size() * 2), dout_split(hx.size() * 2);
        DevBuf dst_full(3 * C * 4), dst_pool(2 * 3 * C * 4), dvpool(2 * T_ver * 3 * C * 4);
        DevBuf dlanes(4), dacc(4);
        dx.upload(hx);
        dw.upload(hw);
        dres.upload(hres);
        dst_full.zero();
        dst_pool.zero();
        dvpool.zero();
        std::vector<std::int32_t> hlane = {1}, hacc = {3};
        dlanes.upload(hlane);
        dacc.upload(hacc);

        // Full prefill
        Tensor tx_all   = make_t(dx.ptr, DType::BF16, C, T_prefill + T_ver);
        Tensor tw       = make_t(dw.ptr, DType::BF16, 4, C);
        Tensor tres_all = make_t(dres.ptr, DType::BF16, C, T_prefill + T_ver);
        Tensor to_all   = make_t(dout_full.ptr, DType::BF16, C, T_prefill + T_ver);
        ops::inkling2_sconv_static_prefill(tx_all, tw, static_cast<float*>(dst_full.ptr), to_all,
                                           &tres_all, stream);

        // Split: prefill T_prefill on lane 1, then verify T_ver on lane 1, then commit acc=3
        Tensor tx_pre   = make_t(dx.ptr, DType::BF16, C, T_prefill);
        Tensor tres_pre = make_t(dres.ptr, DType::BF16, C, T_prefill);
        Tensor to_pre   = make_t(dout_split.ptr, DType::BF16, C, T_prefill);
        ops::inkling2_sconv_static_prefill(
            tx_pre, tw, static_cast<float*>(dst_pool.ptr) + 1 * 3 * C, to_pre, &tres_pre, stream);

        Tensor tx_ver = make_t(static_cast<char*>(dx.ptr) + T_prefill * C * 2, DType::BF16, C, T_ver);
        Tensor tres_ver =
            make_t(static_cast<char*>(dres.ptr) + T_prefill * C * 2, DType::BF16, C, T_ver);
        Tensor to_ver =
            make_t(static_cast<char*>(dout_split.ptr) + T_prefill * C * 2, DType::BF16, C, T_ver);
        ops::inkling2_sconv_static_verify(
            tx_ver, tw, static_cast<const float*>(dst_pool.ptr), static_cast<float*>(dvpool.ptr),
            static_cast<const std::int32_t*>(dlanes.ptr), 1, T_ver, to_ver, &tres_ver, stream,
            3 * C, T_ver);

        ops::inkling2_sconv_commit_verify(static_cast<float*>(dst_pool.ptr),
                                          static_cast<const float*>(dvpool.ptr), 3 * C, T_ver,
                                          static_cast<const std::int32_t*>(dlanes.ptr),
                                          static_cast<const std::int32_t*>(dacc.ptr), 1, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto gfull  = to_f64_from_bf16(dout_full.download<std::uint16_t>(hx.size()));
        auto gsplit = to_f64_from_bf16(dout_split.download<std::uint16_t>(hx.size()));
        check_close("sconv_static prefill == prefill+verify", gfull, gsplit, 0.0, 0.0);

        // Check committed state after acc=3 equals last 3 inputs [T_prefill+0, T_prefill+1, T_prefill+2]
        auto st = dst_pool.download<float>(2 * 3 * C);
        for (int lag = 0; lag < 3; ++lag) {
            const int src_t = T_prefill + lag;
            for (int c = 0; c < C; ++c) {
                const float want = bf16_to_f32(hx[src_t * C + c]);
                const float got  = st[1 * 3 * C + lag * C + c];
                if (std::abs(want - got) > 1e-6f) {
                    throw std::runtime_error("sconv_commit_verify state mismatch");
                }
            }
        }
        std::printf("  [PASS] sconv_static + commit_verify exact state check\n");
    }

    // Dynamic sconv against FP64 oracle
    {
        auto hdyn = rand_bf16(rng, T_prefill * Hkv * 4, 0.4f);
        DevBuf dx(T_prefill * C * 2), hdw_b(hdyn.size() * 2), hsw_b(C * 4 * 2),
            dst(3 * C * 4), dout(T_prefill * C * 2);
        dx.upload(std::vector<std::uint16_t>(hx.begin(), hx.begin() + T_prefill * C));
        hdw_b.upload(hdyn);
        hsw_b.upload(hw);
        dst.zero();
        Tensor tx  = make_t(dx.ptr, DType::BF16, C, T_prefill);
        Tensor tdw = make_t(hdw_b.ptr, DType::BF16, 4 * Hkv, T_prefill);
        Tensor tsw = make_t(hsw_b.ptr, DType::BF16, 4, C);
        Tensor to  = make_t(dout.ptr, DType::BF16, C, T_prefill);
        ops::inkling2_sconv_dynamic_prefill(tx, tdw, tsw, Hkv, static_cast<float*>(dst.ptr), to,
                                            stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> ref(T_prefill * C, 0.0);
        const int hd = C / Hkv;
        for (int t = 0; t < T_prefill; ++t) {
            for (int c = 0; c < C; ++c) {
                const int h = c / hd;
                double acc  = 0.0;
                for (int lag = 0; lag < 4; ++lag) {
                    double x_lag = (t - lag >= 0) ? bf16_to_f32(hx[(t - lag) * C + c]) : 0.0;
                    double w     = bf16_to_f32(hw[c * 4 + lag]) +
                                   bf16_to_f32(hdyn[(t * Hkv + h) * 4 + lag]);
                    acc += w * x_lag;
                }
                ref[t * C + c] = (acc + bf16_to_f32(hx[t * C + c])) / 3.0;
            }
        }
        check_close("sconv_dynamic prefill vs FP64", ref,
                    to_f64_from_bf16(dout.download<std::uint16_t>(T_prefill * C)), 1.5e-2, 1.5e-2);
    }

    // Fused K+V pair sconv (static + dynamic decode & verify match separate calls bit-for-bit!)
    {
        const int B = 2, V = 4;
        auto hk = rand_bf16(rng, B * V * C, 0.7f), hv = rand_bf16(rng, B * V * C, 0.7f);
        auto hwk = rand_bf16(rng, C * 4, 0.5f), hwv = rand_bf16(rng, C * 4, 0.5f);
        auto hdk = rand_bf16(rng, B * V * Hkv * 4, 0.3f), hdv = rand_bf16(rng, B * V * Hkv * 4, 0.3f);
        std::vector<std::int32_t> hlanes = {0, 1};
        const std::size_t stride = 6 * C;
        DevBuf dk(hk.size() * 2), dv(hv.size() * 2), dwk(hwk.size() * 2), dwv(hwv.size() * 2),
            ddk(hdk.size() * 2), ddv(hdv.size() * 2), dlanes(B * 4);
        DevBuf dst_ref(B * stride * 4), dst_pair(B * stride * 4), dv_ref(B * V * stride * 4), dv_pair(B * V * stride * 4);
        DevBuf dok_ref(hk.size() * 2), dov_ref(hv.size() * 2), dok_pair(hk.size() * 2), dov_pair(hv.size() * 2);
        dk.upload(hk); dv.upload(hv); dwk.upload(hwk); dwv.upload(hwv); ddk.upload(hdk); ddv.upload(hdv); dlanes.upload(hlanes);
        dst_ref.zero(); dst_pair.zero(); dv_ref.zero(); dv_pair.zero();

        Tensor tk = make_t(dk.ptr, DType::BF16, C, B * V), tv = make_t(dv.ptr, DType::BF16, C, B * V);
        Tensor twk = make_t(dwk.ptr, DType::BF16, 4, C), twv = make_t(dwv.ptr, DType::BF16, 4, C);
        Tensor tdk = make_t(ddk.ptr, DType::BF16, 4 * Hkv, B * V), tdv = make_t(ddv.ptr, DType::BF16, 4 * Hkv, B * V);
        Tensor tok_r = make_t(dok_ref.ptr, DType::BF16, C, B * V), tov_r = make_t(dov_ref.ptr, DType::BF16, C, B * V);
        Tensor tok_p = make_t(dok_pair.ptr, DType::BF16, C, B * V), tov_p = make_t(dov_pair.ptr, DType::BF16, C, B * V);
        const auto* lp = static_cast<const std::int32_t*>(dlanes.ptr);

        ops::inkling2_sconv_static_verify(tk, twk, static_cast<float*>(dst_ref.ptr), static_cast<float*>(dv_ref.ptr), lp, B, V, tok_r, nullptr, stream, stride, V);
        ops::inkling2_sconv_static_verify(tv, twv, static_cast<float*>(dst_ref.ptr) + 3 * C, static_cast<float*>(dv_ref.ptr) + 3 * C, lp, B, V, tov_r, nullptr, stream, stride, V);
        ops::inkling2_sconv_kv_static_verify(tk, twk, static_cast<float*>(dst_pair.ptr), static_cast<float*>(dv_pair.ptr), tv, twv, static_cast<float*>(dst_pair.ptr) + 3 * C, static_cast<float*>(dv_pair.ptr) + 3 * C, lp, B, V, tok_p, tov_p, stream, stride, V);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close("sconv_kv_static_verify K exact", to_f64_from_bf16(dok_ref.download<std::uint16_t>(hk.size())), to_f64_from_bf16(dok_pair.download<std::uint16_t>(hk.size())), 0.0, 0.0);
        check_close("sconv_kv_static_verify V exact", to_f64_from_bf16(dov_ref.download<std::uint16_t>(hv.size())), to_f64_from_bf16(dov_pair.download<std::uint16_t>(hv.size())), 0.0, 0.0);

        ops::inkling2_sconv_dynamic_verify(tk, tdk, twk, Hkv, static_cast<float*>(dst_ref.ptr), static_cast<float*>(dv_ref.ptr), lp, B, V, tok_r, stream, stride, V);
        ops::inkling2_sconv_dynamic_verify(tv, tdv, twv, Hkv, static_cast<float*>(dst_ref.ptr) + 3 * C, static_cast<float*>(dv_ref.ptr) + 3 * C, lp, B, V, tov_r, stream, stride, V);
        ops::inkling2_sconv_kv_dynamic_verify(tk, tdk, twk, static_cast<float*>(dst_pair.ptr), static_cast<float*>(dv_pair.ptr), tv, tdv, twv, static_cast<float*>(dst_pair.ptr) + 3 * C, static_cast<float*>(dv_pair.ptr) + 3 * C, Hkv, lp, B, V, tok_p, tov_p, stream, stride, V);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close("sconv_kv_dynamic_verify K exact", to_f64_from_bf16(dok_ref.download<std::uint16_t>(hk.size())), to_f64_from_bf16(dok_pair.download<std::uint16_t>(hk.size())), 0.0, 0.0);
        check_close("sconv_kv_dynamic_verify V exact", to_f64_from_bf16(dov_ref.download<std::uint16_t>(hv.size())), to_f64_from_bf16(dov_pair.download<std::uint16_t>(hv.size())), 0.0, 0.0);
    }

    // Fused TP2 Packet AllReduce + Static SConv (Decode & Verify, BF16 & FP32) + Packet AR Residual
    // Verified via single-GPU two-epoch loopback mailbox (local == peer -> exact bf16(2*x)), 2 consecutive rounds!
    {
        const int Hc = 2304, B = 2, V = 4;
        const std::size_t epoch_bytes = 64 * 1024;
        const std::size_t mbox_bytes  = 2 * epoch_bytes + 4096;
        DevBuf mbox(mbox_bytes);
        CUDA_CHECK(cudaMemsetAsync(mbox.ptr, 0xFF, 2 * epoch_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(mbox.ptr) + 2 * epoch_bytes, 0, 4096, stream));

        auto hw_h   = rand_bf16(rng, Hc * 4, 0.4f);
        auto hres_d = rand_bf16(rng, B * Hc, 0.5f);
        auto hbc_d  = rand_bf16(rng, B * Hc, 0.6f);
        DevBuf dw_h(hw_h.size() * 2), dres_d(hres_d.size() * 2), dbc_src(hbc_d.size() * 2), dbc_dst(hbc_d.size() * 2);
        dw_h.upload(hw_h); dres_d.upload(hres_d); dbc_src.upload(hbc_d);
        std::vector<std::int32_t> hlanes = {0, 1};
        DevBuf dlanes(B * 4); dlanes.upload(hlanes);
        const std::size_t stride = 3 * Hc;
        DevBuf dst_ref(B * stride * 4), dst_pkt(B * stride * 4);
        dst_ref.zero(); dst_pkt.zero();

        Tensor tw_h   = make_t(dw_h.ptr, DType::BF16, 4, Hc);
        Tensor tres_d = make_t(dres_d.ptr, DType::BF16, Hc, B);
        Tensor tbc_s  = make_t(dbc_src.ptr, DType::BF16, Hc, B);
        Tensor tbc_d  = make_t(dbc_dst.ptr, DType::BF16, Hc, B);

        // Decode BF16 & FP32 across 2 consecutive epochs
        for (int ep = 0; ep < 2; ++ep) {
            const bool is_f32 = (ep == 1);
            auto hpart_f32    = rand_f32(rng, B * Hc, 0.4f);
            std::vector<std::uint16_t> hpart_bf16(B * Hc), hsum_bf16(B * Hc);
            for (int i = 0; i < B * Hc; ++i) {
                hpart_bf16[i] = f32_to_bf16(hpart_f32[i]);
                float v       = bf16_to_f32(hpart_bf16[i]);
                hsum_bf16[i]  = f32_to_bf16(v + v);
            }
            DevBuf dpart(is_f32 ? B * Hc * 4 : B * Hc * 2), dsum(B * Hc * 2);
            DevBuf do_ref(B * Hc * 2), do_pkt(B * Hc * 2);
            if (is_f32) { dpart.upload(hpart_f32); } else { dpart.upload(hpart_bf16); }
            dsum.upload(hsum_bf16);
            dbc_dst.zero();
            Tensor tpart = make_t(dpart.ptr, is_f32 ? DType::FP32 : DType::BF16, Hc, B);
            Tensor tsum  = make_t(dsum.ptr, DType::BF16, Hc, B);
            Tensor to_r  = make_t(do_ref.ptr, DType::BF16, Hc, B);
            Tensor to_p  = make_t(do_pkt.ptr, DType::BF16, Hc, B);

            ops::inkling2_sconv_static_decode(tsum, tw_h, static_cast<float*>(dst_ref.ptr),
                                              static_cast<const std::int32_t*>(dlanes.ptr), B, to_r,
                                              &tres_d, stream, stride);
            ops::inkling2_tp2_packet_allreduce_sconv_decode(
                tpart, mbox.ptr, mbox.ptr, epoch_bytes, tw_h, static_cast<float*>(dst_pkt.ptr),
                static_cast<const std::int32_t*>(dlanes.ptr), B, stride, to_p, &tres_d, &tbc_s,
                &tbc_d, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            check_close(is_f32 ? "tp2_packet_ar_sconv_decode FP32 exact"
                               : "tp2_packet_ar_sconv_decode BF16 exact",
                        to_f64_from_bf16(do_ref.download<std::uint16_t>(B * Hc)),
                        to_f64_from_bf16(do_pkt.download<std::uint16_t>(B * Hc)), 0.0, 0.0);
            if (dbc_dst.download<std::uint16_t>(B * Hc) != hbc_d) {
                throw std::runtime_error("tp2_packet_ar_sconv_decode block_copy mismatch");
            }
        }

        // Verify BF16 & FP32 across 2 consecutive epochs
        auto hres_v = rand_bf16(rng, B * V * Hc, 0.5f);
        auto hbc_v  = rand_bf16(rng, B * V * Hc, 0.6f);
        DevBuf dres_v(hres_v.size() * 2), dbc_vs(hbc_v.size() * 2), dbc_vd(hbc_v.size() * 2);
        dres_v.upload(hres_v); dbc_vs.upload(hbc_v);
        DevBuf dv_ref(B * V * stride * 4), dv_pkt(B * V * stride * 4);
        Tensor tres_v = make_t(dres_v.ptr, DType::BF16, Hc, B * V);
        Tensor tbv_s  = make_t(dbc_vs.ptr, DType::BF16, Hc, B * V);
        Tensor tbv_d  = make_t(dbc_vd.ptr, DType::BF16, Hc, B * V);
        for (int ep = 0; ep < 2; ++ep) {
            const bool is_f32 = (ep == 1);
            auto hpart_f32    = rand_f32(rng, B * V * Hc, 0.4f);
            std::vector<std::uint16_t> hpart_bf16(B * V * Hc), hsum_bf16(B * V * Hc);
            for (int i = 0; i < B * V * Hc; ++i) {
                hpart_bf16[i] = f32_to_bf16(hpart_f32[i]);
                float v       = bf16_to_f32(hpart_bf16[i]);
                hsum_bf16[i]  = f32_to_bf16(v + v);
            }
            DevBuf dpart(is_f32 ? B * V * Hc * 4 : B * V * Hc * 2), dsum(B * V * Hc * 2);
            DevBuf do_ref(B * V * Hc * 2), do_pkt(B * V * Hc * 2);
            if (is_f32) { dpart.upload(hpart_f32); } else { dpart.upload(hpart_bf16); }
            dsum.upload(hsum_bf16);
            dv_ref.zero(); dv_pkt.zero(); dbc_vd.zero();
            Tensor tpart = make_t(dpart.ptr, is_f32 ? DType::FP32 : DType::BF16, Hc, B * V);
            Tensor tsum  = make_t(dsum.ptr, DType::BF16, Hc, B * V);
            Tensor to_r  = make_t(do_ref.ptr, DType::BF16, Hc, B * V);
            Tensor to_p  = make_t(do_pkt.ptr, DType::BF16, Hc, B * V);

            ops::inkling2_sconv_static_verify(tsum, tw_h, static_cast<const float*>(dst_ref.ptr),
                                              static_cast<float*>(dv_ref.ptr),
                                              static_cast<const std::int32_t*>(dlanes.ptr), B, V,
                                              to_r, &tres_v, stream, stride, V);
            ops::inkling2_tp2_packet_allreduce_sconv_verify(
                tpart, mbox.ptr, mbox.ptr, epoch_bytes, tw_h, static_cast<const float*>(dst_pkt.ptr),
                static_cast<float*>(dv_pkt.ptr), static_cast<const std::int32_t*>(dlanes.ptr), B, V,
                stride, V, to_p, &tres_v, &tbv_s, &tbv_d, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            check_close(is_f32 ? "tp2_packet_ar_sconv_verify FP32"
                               : "tp2_packet_ar_sconv_verify BF16",
                        to_f64_from_bf16(do_ref.download<std::uint16_t>(B * V * Hc)),
                        to_f64_from_bf16(do_pkt.download<std::uint16_t>(B * V * Hc)), 1e-3, 1e-2);
            check_close("tp2_packet_ar_sconv_verify state exact",
                        to_f64_from_f32(dv_ref.download<float>(B * V * stride)),
                        to_f64_from_f32(dv_pkt.download<float>(B * V * stride)), 0.0, 0.0);
            if (dbc_vd.download<std::uint16_t>(B * V * Hc) != hbc_v) {
                throw std::runtime_error("tp2_packet_ar_sconv_verify block_copy mismatch");
            }
        }

        // Wave 4 fused TP2 Packet AllReduce + in-place residual_add (2 consecutive epochs)
        for (int ep = 0; ep < 2; ++ep) {
            auto hp  = rand_bf16(rng, 4 * Hc, 0.5f);
            auto hri = rand_bf16(rng, 4 * Hc, 0.5f);
            DevBuf dp(hp.size() * 2), dri(hri.size() * 2);
            dp.upload(hp); dri.upload(hri);
            Tensor tp  = make_t(dp.ptr, DType::BF16, Hc, 4);
            Tensor tri = make_t(dri.ptr, DType::BF16, Hc, 4);
            ops::inkling2_tp2_packet_allreduce_residual(tp, mbox.ptr, mbox.ptr, epoch_bytes, tri, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<double> ref(4 * Hc);
            for (int i = 0; i < 4 * Hc; ++i) {
                float p    = bf16_to_f32(hp[i]);
                float ar   = bf16_to_f32(f32_to_bf16(p + p));
                ref[i]     = bf16_to_f32(f32_to_bf16(bf16_to_f32(hri[i]) + ar));
            }
            check_close(ep == 0 ? "tp2_packet_ar_residual ep0 exact"
                                : "tp2_packet_ar_residual ep1 exact",
                        ref, to_f64_from_bf16(dri.download<std::uint16_t>(4 * Hc)), 0.0, 0.0);
        }
    }
}

// ============================================================================
// Test 4: Attention Prologue + Prefill + Decode/Verify (both SWA ring and Paged Global)
// ============================================================================
void test_attention(cudaStream_t stream) {
    std::mt19937 rng(303);
    const int n_q = 12, n_kv = 4, D = 128, d_rel = 16, rel_ext = 1024;
    const int packed = 2 * n_q * D + n_q * d_rel + n_q; // 3276
    const int S_pre = 18, S_dec = 4, S_tot = S_pre + S_dec;

    auto hqgrs = rand_bf16(rng, S_tot * packed, 0.5f);
    auto hk    = rand_bf16(rng, S_tot * n_kv * D, 0.5f);
    auto hv    = rand_bf16(rng, S_tot * n_kv * D, 0.5f);
    auto hqs   = rand_bf16(rng, n_q * D, 0.8f);
    auto hrs   = rand_bf16(rng, n_q * d_rel, 0.8f);
    auto hrp   = rand_bf16(rng, d_rel * rel_ext, 0.2f);

    for (bool is_local : {true, false}) {
        const int swa_win = 10; // small window to exercise SWA cutoff on S_tot=22!
        DevBuf dqgrs(hqgrs.size() * 2), dk(hk.size() * 2), dv(hv.size() * 2);
        DevBuf dqs(hqs.size() * 2), drs(hrs.size() * 2), drp(hrp.size() * 2);
        DevBuf dqout(S_tot * n_q * D * 2), drbout(S_tot * n_q * rel_ext * 4);
        DevBuf dout_pre(S_tot * n_q * D * 2), dout_ver(S_dec * n_q * D * 2);
        DevBuf dsplit(S_dec * n_q * 16 * (D + 4) * 4);
        dqgrs.upload(hqgrs);
        dk.upload(hk);
        dv.upload(hv);
        dqs.upload(hqs);
        drs.upload(hrs);
        drp.upload(hrp);

        // KV storage
        const int ring_cap = 32;
        const int max_pages = 4;
        DevBuf dring_k(ring_cap * n_kv * D * 2), dring_v(ring_cap * n_kv * D * 2);
        DevBuf dpage_k(max_pages * n_kv * 64 * D * 2), dpage_v(max_pages * n_kv * 64 * D * 2);
        DevBuf dbt(max_pages * 4);
        std::vector<std::int32_t> hbt = {2, 0, 3, 1}; // non-trivial page permutation!
        dbt.upload(hbt);

        ops::Inkling2KvTarget kv{};
        kv.is_local          = is_local;
        kv.ring_k            = dring_k.ptr;
        kv.ring_v            = dring_v.ptr;
        kv.ring_cap          = ring_cap;
        kv.paged_k           = dpage_k.ptr;
        kv.paged_v           = dpage_v.ptr;
        kv.block_tables      = static_cast<const std::int32_t*>(dbt.ptr);
        kv.max_logical_pages = max_pages;

        std::vector<std::int32_t> hpos(S_tot), hlanes(S_tot, 0);
        for (int i = 0; i < S_tot; ++i) { hpos[i] = i; }
        DevBuf dpos(S_tot * 4), dlanes(S_tot * 4);
        dpos.upload(hpos);
        dlanes.upload(hlanes);

        Tensor tqgrs = make_t(dqgrs.ptr, DType::BF16, packed, S_tot);
        Tensor tk    = make_t(dk.ptr, DType::BF16, n_kv * D, S_tot);
        Tensor tv    = make_t(dv.ptr, DType::BF16, n_kv * D, S_tot);
        Tensor tqs   = make_t(dqs.ptr, DType::BF16, D, n_q);
        Tensor trs   = make_t(drs.ptr, DType::BF16, d_rel, n_q);
        Tensor trp   = make_t(drp.ptr, DType::BF16, rel_ext, d_rel);
        Tensor tqout = make_t(dqout.ptr, DType::BF16, D, n_q, S_tot);
        Tensor trb   = make_t(drbout.ptr, DType::FP32, rel_ext, n_q, S_tot);

        ops::inkling2_attn_prologue(tqgrs, tk, tv, tqs, trs, trp,
                                    static_cast<const std::int32_t*>(dpos.ptr),
                                    static_cast<const std::int32_t*>(dlanes.ptr), S_tot, n_q, n_kv,
                                    kv, is_local, 8.0f, 8.0f, 8.0f, 0.25f, 1e-6f, tqout, trb,
                                    stream);

        // Run prefill over all S_tot tokens
        Tensor top = make_t(dout_pre.ptr, DType::BF16, n_q * D, S_tot);
        ops::inkling2_attn_prefill(tqout, tqgrs, trb, kv, 0, 0, S_tot, n_q, n_kv, swa_win, top,
                                   stream);

        // Run decode_verify over last S_dec tokens
        Tensor tq_sub =
            make_t(static_cast<char*>(dqout.ptr) + S_pre * n_q * D * 2, DType::BF16, D, n_q, S_dec);
        Tensor tqgrs_sub = make_t(static_cast<char*>(dqgrs.ptr) + S_pre * packed * 2, DType::BF16,
                                  packed, S_dec);
        Tensor trb_sub = make_t(static_cast<char*>(drbout.ptr) + S_pre * n_q * rel_ext * 4,
                                DType::FP32, rel_ext, n_q, S_dec);
        Tensor tov     = make_t(dout_ver.ptr, DType::BF16, n_q * D, S_dec);
        ops::inkling2_attn_decode_verify(
            tq_sub, tqgrs_sub, trb_sub, kv, static_cast<const std::int32_t*>(dlanes.ptr),
            static_cast<const std::int32_t*>(dpos.ptr) + S_pre, 1, S_dec, n_q, n_kv, swa_win,
            static_cast<float*>(dsplit.ptr), 16, tov, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto gpre = to_f64_from_bf16(dout_pre.download<std::uint16_t>(S_tot * n_q * D));
        auto gver = to_f64_from_bf16(dout_ver.download<std::uint16_t>(S_dec * n_q * D));
        std::vector<double> gpre_tail(gpre.begin() + S_pre * n_q * D, gpre.end());
        check_close(is_local ? "attn SWA prefill == decode_verify" : "attn Full prefill == decode_verify",
                    gpre_tail, gver, 1.5e-2, 1.5e-2);

        // Independent FP64 oracle for last token (pos = S_tot - 1)
        const int t_check = S_tot - 1;
        auto gq_all       = dout_pre.download<std::uint16_t>(S_tot * n_q * D); // unused, read qout
        auto gq_d         = dqout.download<std::uint16_t>(S_tot * n_q * D);
        auto grb_d        = drbout.download<float>(S_tot * n_q * rel_ext);

        // Compute K_clipped, V_normed in FP64 from original hk/hv
        std::vector<double> k_f64(S_tot * n_kv * D), v_f64(S_tot * n_kv * D);
        for (int t = 0; t < S_tot; ++t) {
            for (int h = 0; h < n_kv; ++h) {
                double sk = 0, sv = 0;
                for (int d = 0; d < D; ++d) {
                    double kv0 = bf16_to_f32(hk[(t * n_kv + h) * D + d]);
                    double vv0 = bf16_to_f32(hv[(t * n_kv + h) * D + d]);
                    sk += kv0 * kv0;
                    sv += vv0 * vv0;
                }
                double kf = std::min(1.0, 8.0 / (std::sqrt(sk) + 1e-6));
                double vf = 1.0 / std::sqrt(sv / D + 1e-6);
                for (int d = 0; d < D; ++d) {
                    k_f64[(t * n_kv + h) * D + d] = bf16_to_f32(hk[(t * n_kv + h) * D + d]) * kf;
                    v_f64[(t * n_kv + h) * D + d] = bf16_to_f32(hv[(t * n_kv + h) * D + d]) * vf;
                }
            }
        }
        std::vector<double> ref_last(n_q * D);
        const int k_start = is_local ? std::max(0, t_check - swa_win + 1) : 0;
        for (int qh = 0; qh < n_q; ++qh) {
            const int kvh  = qh / 3;
            const double sink =
                bf16_to_f32(hqgrs[t_check * packed + 2 * n_q * D + n_q * d_rel + qh]);
            std::vector<double> scores;
            double mx = sink;
            for (int kp = k_start; kp <= t_check; ++kp) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += bf16_to_f32(gq_d[(t_check * n_q + qh) * D + d]) *
                           k_f64[(kp * n_kv + kvh) * D + d];
                }
                dot /= std::sqrt(static_cast<double>(D));
                const int dist = t_check - kp;
                if (dist < rel_ext) { dot += grb_d[(t_check * n_q + qh) * rel_ext + dist]; }
                scores.push_back(dot);
                mx = std::max(mx, dot);
            }
            double denom = std::exp(sink - mx);
            std::vector<double> acc(D, 0.0);
            for (std::size_t idx = 0; idx < scores.size(); ++idx) {
                const int kp = k_start + static_cast<int>(idx);
                double p     = std::exp(scores[idx] - mx);
                denom += p;
                for (int d = 0; d < D; ++d) { acc[d] += p * v_f64[(kp * n_kv + kvh) * D + d]; }
            }
            for (int d = 0; d < D; ++d) {
                double g = bf16_to_f32(hqgrs[t_check * packed + n_q * D + qh * D + d]);
                ref_last[qh * D + d] = (acc[d] / denom) / (1.0 + std::exp(-g));
            }
        }
        std::vector<double> got_last(gver.begin() + (S_dec - 1) * n_q * D, gver.end());
        check_close(is_local ? "attn SWA decode_verify vs FP64 oracle"
                             : "attn Full decode_verify vs FP64 oracle",
                    ref_last, got_last, 2e-2, 2e-2);

        // Also test Wave 4 single-launch adaptive split_done at short context (eff_splits == 1)
        DevBuf dsdone(S_dec * n_q * 4);
        dsdone.zero();
        DevBuf dout_ad(S_dec * n_q * D * 2);
        Tensor to_ad = make_t(dout_ad.ptr, DType::BF16, n_q * D, S_dec);
        ops::inkling2_attn_decode_verify(
            tq_sub, tqgrs_sub, trb_sub, kv, static_cast<const std::int32_t*>(dlanes.ptr),
            static_cast<const std::int32_t*>(dpos.ptr) + S_pre, 1, S_dec, n_q, n_kv, swa_win,
            static_cast<float*>(dsplit.ptr), 16, to_ad, stream,
            static_cast<std::int32_t*>(dsdone.ptr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close(is_local ? "attn SWA adaptive (eff_splits=1) vs FP64"
                             : "attn Full adaptive (eff_splits=1) vs FP64",
                    ref_last,
                    [&]() {
                        auto v = to_f64_from_bf16(dout_ad.download<std::uint16_t>(S_dec * n_q * D));
                        return std::vector<double>(v.begin() + (S_dec - 1) * n_q * D, v.end());
                    }(),
                    2e-2, 2e-2);
    }

    // Fused K/V SConv inside attn_prologue (static & dynamic x decode & verify) vs separate sconv + prologue
    for (bool is_dyn : {false, true}) {
        for (bool is_ver : {false, true}) {
            const int B = 2, Q = is_ver ? 4 : 1, T = B * Q, C = n_kv * D;
            const std::size_t stride = 6 * C;
            auto hqgrs_s = rand_bf16(rng, T * packed, 0.5f);
            auto hk_s    = rand_bf16(rng, T * C, 0.5f);
            auto hv_s    = rand_bf16(rng, T * C, 0.5f);
            auto hwk     = rand_bf16(rng, C * 4, 0.4f);
            auto hwv     = rand_bf16(rng, C * 4, 0.4f);
            auto hdk     = rand_bf16(rng, T * n_kv * 4, 0.3f);
            auto hdv     = rand_bf16(rng, T * n_kv * 4, 0.3f);
            auto hst0    = rand_f32(rng, B * stride, 0.4f);
            std::vector<std::int32_t> hlanes_b = {0, 1}, hlanes_t(T), hpos_t(T);
            for (int b = 0; b < B; ++b) {
                for (int q = 0; q < Q; ++q) {
                    hlanes_t[b * Q + q] = b;
                    hpos_t[b * Q + q]   = 10 + q;
                }
            }
            const int ring_cap = 32;
            DevBuf dqgrs(hqgrs_s.size() * 2), dk(hk_s.size() * 2), dv(hv_s.size() * 2);
            DevBuf dqs(hqs.size() * 2), drs(hrs.size() * 2), drp(hrp.size() * 2);
            DevBuf dwk(hwk.size() * 2), dwv(hwv.size() * 2), ddk(hdk.size() * 2), ddv(hdv.size() * 2);
            DevBuf dst_ref(B * stride * 4), dst_fus(B * stride * 4);
            DevBuf dv_ref(B * Q * stride * 4), dv_fus(B * Q * stride * 4);
            DevBuf dlb(B * 4), dlt(T * 4), dpt(T * 4);
            DevBuf dk_sc(hk_s.size() * 2), dv_sc(hv_s.size() * 2);
            DevBuf dqo_r(T * n_q * D * 2), drb_r(T * n_q * rel_ext * 4);
            DevBuf dqo_f(T * n_q * D * 2), drb_f(T * n_q * rel_ext * 4);
            DevBuf drk_r(B * ring_cap * C * 2), drv_r(B * ring_cap * C * 2);
            DevBuf drk_f(B * ring_cap * C * 2), drv_f(B * ring_cap * C * 2);
            dqgrs.upload(hqgrs_s); dk.upload(hk_s); dv.upload(hv_s);
            dqs.upload(hqs); drs.upload(hrs); drp.upload(hrp);
            dwk.upload(hwk); dwv.upload(hwv); ddk.upload(hdk); ddv.upload(hdv);
            dst_ref.upload(hst0); dst_fus.upload(hst0);
            dv_ref.zero(); dv_fus.zero(); drk_r.zero(); drv_r.zero(); drk_f.zero(); drv_f.zero();
            dlb.upload(hlanes_b); dlt.upload(hlanes_t); dpt.upload(hpos_t);

            Tensor tqgrs = make_t(dqgrs.ptr, DType::BF16, packed, T);
            Tensor tk    = make_t(dk.ptr, DType::BF16, C, T), tv = make_t(dv.ptr, DType::BF16, C, T);
            Tensor tqs   = make_t(dqs.ptr, DType::BF16, D, n_q), trs = make_t(drs.ptr, DType::BF16, d_rel, n_q);
            Tensor trp   = make_t(drp.ptr, DType::BF16, rel_ext, d_rel);
            Tensor twk   = make_t(dwk.ptr, DType::BF16, 4, C), twv = make_t(dwv.ptr, DType::BF16, 4, C);
            Tensor tdk   = make_t(ddk.ptr, DType::BF16, 4 * n_kv, T), tdv = make_t(ddv.ptr, DType::BF16, 4 * n_kv, T);
            Tensor tksc  = make_t(dk_sc.ptr, DType::BF16, C, T), tvsc = make_t(dv_sc.ptr, DType::BF16, C, T);
            Tensor tqo_r = make_t(dqo_r.ptr, DType::BF16, D, n_q, T), trb_r = make_t(drb_r.ptr, DType::FP32, rel_ext, n_q, T);
            Tensor tqo_f = make_t(dqo_f.ptr, DType::BF16, D, n_q, T), trb_f = make_t(drb_f.ptr, DType::FP32, rel_ext, n_q, T);

            ops::Inkling2KvTarget kv_r{}, kv_f{};
            kv_r.is_local = true; kv_r.ring_k = drk_r.ptr; kv_r.ring_v = drv_r.ptr; kv_r.ring_cap = ring_cap;
            kv_f.is_local = true; kv_f.ring_k = drk_f.ptr; kv_f.ring_v = drv_f.ptr; kv_f.ring_cap = ring_cap;

            // Unfused reference
            if (!is_dyn && !is_ver) {
                ops::inkling2_sconv_kv_static_decode(tk, twk, static_cast<float*>(dst_ref.ptr),
                                                     tv, twv, static_cast<float*>(dst_ref.ptr) + 3 * C,
                                                     static_cast<const std::int32_t*>(dlb.ptr), B,
                                                     tksc, tvsc, stream, stride);
            } else if (!is_dyn && is_ver) {
                ops::inkling2_sconv_kv_static_verify(tk, twk, static_cast<const float*>(dst_ref.ptr),
                                                     static_cast<float*>(dv_ref.ptr),
                                                     tv, twv, static_cast<const float*>(dst_ref.ptr) + 3 * C,
                                                     static_cast<float*>(dv_ref.ptr) + 3 * C,
                                                     static_cast<const std::int32_t*>(dlb.ptr), B, Q,
                                                     tksc, tvsc, stream, stride, Q);
            } else if (is_dyn && !is_ver) {
                ops::inkling2_sconv_kv_dynamic_decode(tk, tdk, twk, static_cast<float*>(dst_ref.ptr),
                                                      tv, tdv, twv, static_cast<float*>(dst_ref.ptr) + 3 * C,
                                                      n_kv, static_cast<const std::int32_t*>(dlb.ptr), B,
                                                      tksc, tvsc, stream, stride);
            } else {
                ops::inkling2_sconv_kv_dynamic_verify(tk, tdk, twk, static_cast<const float*>(dst_ref.ptr),
                                                      static_cast<float*>(dv_ref.ptr),
                                                      tv, tdv, twv, static_cast<const float*>(dst_ref.ptr) + 3 * C,
                                                      static_cast<float*>(dv_ref.ptr) + 3 * C,
                                                      n_kv, static_cast<const std::int32_t*>(dlb.ptr), B, Q,
                                                      tksc, tvsc, stream, stride, Q);
            }
            ops::inkling2_attn_prologue(tqgrs, tksc, tvsc, tqs, trs, trp,
                                        static_cast<const std::int32_t*>(dpt.ptr),
                                        static_cast<const std::int32_t*>(dlt.ptr), T, n_q, n_kv,
                                        kv_r, !is_dyn, 8.0f, 8.0f, 8.0f, 0.25f, 1e-6f, tqo_r, trb_r,
                                        stream);

            // Fused prologue
            ops::Inkling2AttnSconvArgs sc{};
            sc.is_dynamic     = is_dyn;
            sc.k_weight       = dwk.ptr;
            sc.v_weight       = dwv.ptr;
            sc.k_dyn          = is_dyn ? ddk.ptr : nullptr;
            sc.v_dyn          = is_dyn ? ddv.ptr : nullptr;
            sc.k_state_pool   = static_cast<float*>(dst_fus.ptr);
            sc.v_state_pool   = static_cast<float*>(dst_fus.ptr) + 3 * C;
            sc.k_verify_pool  = is_ver ? static_cast<float*>(dv_fus.ptr) : nullptr;
            sc.v_verify_pool  = is_ver ? (static_cast<float*>(dv_fus.ptr) + 3 * C) : nullptr;
            sc.lane_stride    = stride;
            sc.q_per_lane     = Q;
            sc.max_verify_len = Q;
            ops::inkling2_attn_prologue(tqgrs, tk, tv, tqs, trs, trp,
                                        static_cast<const std::int32_t*>(dpt.ptr),
                                        static_cast<const std::int32_t*>(dlt.ptr), T, n_q, n_kv,
                                        kv_f, !is_dyn, 8.0f, 8.0f, 8.0f, 0.25f, 1e-6f, tqo_f, trb_f,
                                        stream, &sc);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "attn_prologue+sconv %s %s KV", is_dyn ? "dyn" : "stat",
                          is_ver ? "ver" : "dec");
            check_close(lbl, to_f64_from_bf16(drk_r.download<std::uint16_t>(B * ring_cap * C)),
                        to_f64_from_bf16(drk_f.download<std::uint16_t>(B * ring_cap * C)), 1e-2, 1e-2);
            check_close(lbl, to_f64_from_bf16(drv_r.download<std::uint16_t>(B * ring_cap * C)),
                        to_f64_from_bf16(drv_f.download<std::uint16_t>(B * ring_cap * C)), 1e-2, 1e-2);
            check_close("attn_prologue+sconv state",
                        to_f64_from_f32(dst_ref.download<float>(B * stride)),
                        to_f64_from_f32(dst_fus.download<float>(B * stride)), 1e-5, 1e-5);
            if (is_ver) {
                check_close("attn_prologue+sconv verify_state",
                            to_f64_from_f32(dv_ref.download<float>(B * Q * stride)),
                            to_f64_from_f32(dv_fus.download<float>(B * Q * stride)), 1e-5, 1e-5);
            }
        }
    }

    // Wave 4 Long-Context Adaptive Last-CTA Split-KV (total_k = 1200 -> eff_splits > 1)
    // Exercises BOTH cyclic SWA ring wrap (ring_cap=1024) AND paged global (permuted pages)!
    // total_k = 4096 (global only) gives >= 160 keys per split, which selects the Wave 5
    // warp-uniform single-exp online softmax route for T >= 2 global attention.
    for (const int total_k : {1200, 4096}) for (bool is_local : {true, false}) {
        if (is_local && total_k != 1200) { continue; }
        const int q_per = 4, swa_win = 1024;
        const int ring_cap = 1024, max_pages = (total_k + 63) / 64 + 4;
        auto hq_l    = rand_bf16(rng, q_per * n_q * D, 0.4f);
        auto hqgrs_l = rand_bf16(rng, q_per * packed, 0.4f);
        auto hrb_l   = rand_f32(rng, q_per * n_q * rel_ext, 0.1f);
        auto hk_all  = rand_bf16(rng, total_k * n_kv * D, 0.35f);
        auto hv_all  = rand_bf16(rng, total_k * n_kv * D, 0.35f);

        std::vector<std::int32_t> hbt(max_pages);
        // Deterministic physical-page permutation (independent of the data RNG stream).
        std::iota(hbt.begin(), hbt.end(), 0);
        std::shuffle(hbt.begin(), hbt.end(), std::mt19937(907));
        std::vector<std::uint16_t> hring_k(ring_cap * n_kv * D, 0), hring_v(ring_cap * n_kv * D, 0);
        std::vector<std::uint16_t> hpage_k(max_pages * n_kv * 64 * D, 0), hpage_v(max_pages * n_kv * 64 * D, 0);
        for (int kp = 0; kp < total_k; ++kp) {
            const int slot = kp % ring_cap;
            const int pp   = hbt[kp >> 6];
            const int ps   = kp & 63;
            for (int h = 0; h < n_kv; ++h) {
                for (int d = 0; d < D; ++d) {
                    const std::uint16_t kv_b = hk_all[(kp * n_kv + h) * D + d];
                    const std::uint16_t vv_b = hv_all[(kp * n_kv + h) * D + d];
                    hring_k[(slot * n_kv + h) * D + d]           = kv_b;
                    hring_v[(slot * n_kv + h) * D + d]           = vv_b;
                    hpage_k[((pp * n_kv + h) * 64 + ps) * D + d] = kv_b;
                    hpage_v[((pp * n_kv + h) * 64 + ps) * D + d] = vv_b;
                }
            }
        }
        std::vector<std::int32_t> hlane = {0}, hpos(q_per);
        for (int i = 0; i < q_per; ++i) { hpos[i] = total_k - q_per + i; }

        DevBuf dq_l(hq_l.size() * 2), dqgrs_l(hqgrs_l.size() * 2), drb_l(hrb_l.size() * 4);
        DevBuf dring_k(hring_k.size() * 2), dring_v(hring_v.size() * 2);
        DevBuf dpage_k(hpage_k.size() * 2), dpage_v(hpage_v.size() * 2);
        DevBuf dbt(max_pages * 4), dl(4), dp(q_per * 4);
        DevBuf dsplit(q_per * n_q * 16 * (D + 4) * 4), dsdone(q_per * n_q * 4), dout(q_per * n_q * D * 2);
        dq_l.upload(hq_l); dqgrs_l.upload(hqgrs_l); drb_l.upload(hrb_l);
        dring_k.upload(hring_k); dring_v.upload(hring_v);
        dpage_k.upload(hpage_k); dpage_v.upload(hpage_v);
        dbt.upload(hbt); dl.upload(hlane); dp.upload(hpos);
        dsdone.zero();

        ops::Inkling2KvTarget kv{};
        kv.is_local          = is_local;
        kv.ring_k            = dring_k.ptr;
        kv.ring_v            = dring_v.ptr;
        kv.ring_cap          = ring_cap;
        kv.paged_k           = dpage_k.ptr;
        kv.paged_v           = dpage_v.ptr;
        kv.block_tables      = static_cast<const std::int32_t*>(dbt.ptr);
        kv.max_logical_pages = max_pages;

        Tensor tq_l    = make_t(dq_l.ptr, DType::BF16, D, n_q, q_per);
        Tensor tqgrs_l = make_t(dqgrs_l.ptr, DType::BF16, packed, q_per);
        Tensor trb_l   = make_t(drb_l.ptr, DType::FP32, rel_ext, n_q, q_per);
        Tensor to_l    = make_t(dout.ptr, DType::BF16, n_q * D, q_per);
        ops::inkling2_attn_decode_verify(tq_l, tqgrs_l, trb_l, kv,
                                         static_cast<const std::int32_t*>(dl.ptr),
                                         static_cast<const std::int32_t*>(dp.ptr), 1, q_per, n_q,
                                         n_kv, swa_win, static_cast<float*>(dsplit.ptr), 16, to_l,
                                         stream, static_cast<std::int32_t*>(dsdone.ptr));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto done_after = dsdone.download<std::int32_t>(q_per * n_q);
        for (int v : done_after) {
            if (v != 0) { throw std::runtime_error("attn adaptive split_done not reset to 0"); }
        }
        // Independent FP64 oracle on last query (pos = total_k - 1)
        const int qi = q_per - 1, qpos = hpos[qi];
        const int k_start = is_local ? std::max(0, qpos - swa_win + 1) : 0;
        std::vector<double> ref(n_q * D);
        for (int qh = 0; qh < n_q; ++qh) {
            const int kvh     = qh / 3;
            const double sink = bf16_to_f32(hqgrs_l[qi * packed + 2 * n_q * D + n_q * d_rel + qh]);
            std::vector<double> scs;
            scs.reserve(qpos - k_start + 1);
            double mx = sink;
            for (int kp = k_start; kp <= qpos; ++kp) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += bf16_to_f32(hq_l[(qi * n_q + qh) * D + d]) *
                           bf16_to_f32(hk_all[(kp * n_kv + kvh) * D + d]);
                }
                dot /= std::sqrt(static_cast<double>(D));
                const int dist = qpos - kp;
                if (dist < rel_ext) { dot += hrb_l[(qi * n_q + qh) * rel_ext + dist]; }
                scs.push_back(dot);
                mx = std::max(mx, dot);
            }
            double den = std::exp(sink - mx);
            std::vector<double> acc(D, 0.0);
            for (std::size_t idx = 0; idx < scs.size(); ++idx) {
                const int kp = k_start + static_cast<int>(idx);
                double p     = std::exp(scs[idx] - mx);
                den += p;
                for (int d = 0; d < D; ++d) {
                    acc[d] += p * bf16_to_f32(hv_all[(kp * n_kv + kvh) * D + d]);
                }
            }
            for (int d = 0; d < D; ++d) {
                double g = bf16_to_f32(hqgrs_l[qi * packed + n_q * D + qh * D + d]);
                ref[qh * D + d] = (acc[d] / den) / (1.0 + std::exp(-g));
            }
        }
        auto got_all = to_f64_from_bf16(dout.download<std::uint16_t>(q_per * n_q * D));
        std::vector<double> got_last(got_all.begin() + qi * n_q * D, got_all.end());
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "attn %s long-ctx adaptive Split-KV (%d) vs FP64",
                      is_local ? "SWA" : "Full", total_k);
        check_close(lbl, ref, got_last, 2e-2, 2e-2);
    }
}

// ============================================================================
// Test 5: Sparse MoE Router + NVFP4 (TP1 & TP2 sum) + MXFP8 (TP1 & TP2 sum)
// ============================================================================
void test_sparse_moe(cudaStream_t stream) {
    std::mt19937 rng(404);
    const int E = 128, H = 2304, I_slab = 576;
    const int max_T = 16;

    // Allocate scratch
    DevBuf d_rlog(max_T * E * 4), d_rbias(E * 4), d_tids(max_T * 8 * 4), d_tw(max_T * 8 * 4);
    DevBuf d_ecnt(E * 4), d_eoff((E + 1) * 4), d_spairs(max_T * 8 * 4), d_aexp(E * 4), d_nact(4);
    DevBuf d_inter(max_T * 8 * 2 * I_slab * 2), d_acc(max_T * H * 4), d_rdone(4);
    d_rdone.zero();

    ops::Inkling2MoeScratch sc{};
    sc.router_logits     = static_cast<float*>(d_rlog.ptr);
    sc.topk_ids          = static_cast<std::int32_t*>(d_tids.ptr);
    sc.topk_weights      = static_cast<float*>(d_tw.ptr);
    sc.expert_counts     = static_cast<std::int32_t*>(d_ecnt.ptr);
    sc.expert_offsets    = static_cast<std::int32_t*>(d_eoff.ptr);
    sc.sorted_pairs      = static_cast<std::int32_t*>(d_spairs.ptr);
    sc.active_experts    = static_cast<std::int32_t*>(d_aexp.ptr);
    sc.num_active        = static_cast<std::int32_t*>(d_nact.ptr);
    sc.intermediate_bf16 = d_inter.ptr;
    sc.accum_f32         = static_cast<float*>(d_acc.ptr);
    sc.router_done       = static_cast<std::int32_t*>(d_rdone.ptr);

    // Test Last-CTA fused router (T=1, 4, 8) against separate bf16_linear + moe_route!
    for (int t_rt : {1, 4, 8}) {
        auto hx = rand_bf16(rng, t_rt * H, 0.4f);
        auto hw = rand_bf16(rng, E * H, 0.3f);
        auto hb = rand_f32(rng, E, 0.2f);
        DevBuf dx(hx.size() * 2), dw(hw.size() * 2);
        dx.upload(hx); dw.upload(hw); d_rbias.upload(hb);
        Tensor tx = make_t(dx.ptr, DType::BF16, H, t_rt);
        Tensor tw = make_t(dw.ptr, DType::BF16, H, E);
        Tensor tb = make_t(d_rbias.ptr, DType::FP32, E);
        Tensor tr = make_t(d_rlog.ptr, DType::FP32, E, t_rt);

        ops::inkling2_bf16_linear(tx, tw, tr, stream);
        ops::inkling2_moe_route(tr, tb, 8.0f, sc, t_rt, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto ref_ids = d_tids.download<std::int32_t>(t_rt * 8);
        auto ref_w   = d_tw.download<float>(t_rt * 8);

        ops::inkling2_moe_router_fused(tx, tw, tb, 8.0f, sc, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto got_ids = d_tids.download<std::int32_t>(t_rt * 8);
        auto got_w   = d_tw.download<float>(t_rt * 8);
        if (ref_ids != got_ids) { throw std::runtime_error("moe_router_fused topk_ids mismatch"); }
        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "moe_router_fused T=%d (Last-CTA exact)", t_rt);
        check_close(lbl, to_f64_from_f32(ref_w), to_f64_from_f32(got_w), 1e-4, 1e-4);
    }

    // Random valid E4M3FN byte generator (avoid NaN 0x7F/0xFF and huge exps)
    auto rand_e4m3 = [&](std::size_t n, int max_exp = 8) {
        std::vector<std::uint8_t> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            int sign = rng() & 1;
            int exp  = (rng() % max_exp) + 1;
            int man  = rng() & 7;
            v[i]     = static_cast<std::uint8_t>((sign << 7) | (exp << 3) | man);
        }
        return v;
    };
    auto rand_e4m3_pos = [&](std::size_t n) {
        std::vector<std::uint8_t> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            int exp = 3 + (rng() % 5); // positive finite scales in [0.0625 .. 1.875]
            int man = rng() & 7;
            v[i]    = static_cast<std::uint8_t>((exp << 3) | man);
        }
        return v;
    };

    // --- NVFP4 MoE (T=1 and T=6, check TP1 vs FP64 AND TP1 == TP2_slab0 + TP2_slab1) ---
    {
        const std::size_t w13c_sz = 2ULL * E * 1152 * 1152;
        const std::size_t w13s_sz = 2ULL * E * 1152 * 144;
        const std::size_t w2c_sz  = 2ULL * E * H * 288;
        const std::size_t w2s_sz  = 2ULL * E * H * 36;
        // Fill random weights (we only need experts 0..15 populated meaningfully, fill all deterministically fast)
        std::vector<std::uint8_t> w13c(w13c_sz, 0), w13s(w13s_sz, 0x38);
        std::vector<std::uint8_t> w2c(w2c_sz, 0), w2s(w2s_sz, 0x38);
        for (int slab = 0; slab < 2; ++slab) {
            for (int e = 0; e < 16; ++e) {
                auto base13c = (static_cast<std::size_t>(slab) * E + e) * 1152 * 1152;
                auto base13s = (static_cast<std::size_t>(slab) * E + e) * 1152 * 144;
                for (std::size_t i = 0; i < 1152 * 1152; ++i) { w13c[base13c + i] = static_cast<std::uint8_t>(rng()); }
                auto s13 = rand_e4m3_pos(1152 * 144);
                std::memcpy(&w13s[base13s], s13.data(), s13.size());
                auto base2c = (static_cast<std::size_t>(slab) * E + e) * H * 288;
                auto base2s = (static_cast<std::size_t>(slab) * E + e) * H * 36;
                for (std::size_t i = 0; i < static_cast<std::size_t>(H) * 288; ++i) {
                    w2c[base2c + i] = static_cast<std::uint8_t>(rng());
                }
                auto s2 = rand_e4m3_pos(static_cast<std::size_t>(H) * 36);
                std::memcpy(&w2s[base2s], s2.data(), s2.size());
            }
        }
        std::vector<float> w13a(E, 0.5f), w2a(E, 0.5f);
        for (int e = 0; e < 16; ++e) {
            w13a[e] = 0.2f + 0.03f * e;
            w2a[e]  = 0.15f + 0.02f * e;
        }
        std::vector<float> w13d = {10.0f}, w2d = {12.0f};

        DevBuf dw13c(w13c_sz), dw13s(w13s_sz), dw13a(E * 4), dw13d(4);
        DevBuf dw2c(w2c_sz), dw2s(w2s_sz), dw2a(E * 4), dw2d(4);
        dw13c.upload(w13c);
        dw13s.upload(w13s);
        dw13a.upload(w13a);
        dw13d.upload(w13d);
        dw2c.upload(w2c);
        dw2s.upload(w2s);
        dw2a.upload(w2a);
        dw2d.upload(w2d);

        for (int T : {1, 6}) {
            auto hx = rand_bf16(rng, T * H, 0.3f);
            // Force router to pick among experts 0..15
            std::vector<float> hrl(T * E, -10.0f);
            std::vector<float> hrb(E, 0.0f);
            for (int t = 0; t < T; ++t) {
                for (int e = 0; e < 16; ++e) { hrl[t * E + e] = rand_f32(rng, 1, 2.0f)[0]; }
            }
            for (int e = 0; e < 16; ++e) { hrb[e] = rand_f32(rng, 1, 0.2f)[0]; }

            DevBuf dx(hx.size() * 2), dout_tp1(T * H * 2), dout_s0(T * H * 2), dout_s1(T * H * 2);
            dx.upload(hx);
            d_rlog.upload(hrl);
            d_rbias.upload(hrb);

            Tensor trl = make_t(d_rlog.ptr, DType::FP32, E, T);
            Tensor trb = make_t(d_rbias.ptr, DType::FP32, E);
            ops::inkling2_moe_route(trl, trb, 8.0f, sc, T, stream);

            Tensor tx   = make_t(dx.ptr, DType::BF16, H, T);
            Tensor to1  = make_t(dout_tp1.ptr, DType::BF16, H, T);
            Tensor tos0 = make_t(dout_s0.ptr, DType::BF16, H, T);
            Tensor tos1 = make_t(dout_s1.ptr, DType::BF16, H, T);

            ops::inkling2_moe_nvfp4_forward(tx, dw13c.ptr, dw13s.ptr,
                                            static_cast<const float*>(dw13a.ptr),
                                            static_cast<const float*>(dw13d.ptr), dw2c.ptr,
                                            dw2s.ptr, static_cast<const float*>(dw2a.ptr),
                                            static_cast<const float*>(dw2d.ptr), 2, 4.0f, 25.0f,
                                            sc, to1, stream);
            // Slab 0 only (rank 0)
            ops::inkling2_moe_nvfp4_forward(tx, dw13c.ptr, dw13s.ptr,
                                            static_cast<const float*>(dw13a.ptr),
                                            static_cast<const float*>(dw13d.ptr), dw2c.ptr,
                                            dw2s.ptr, static_cast<const float*>(dw2a.ptr),
                                            static_cast<const float*>(dw2d.ptr), 1, 4.0f, 25.0f,
                                            sc, tos0, stream);
            // Slab 1 only (rank 1)
            ops::inkling2_moe_nvfp4_forward(
                tx, static_cast<const char*>(dw13c.ptr) + w13c_sz / 2,
                static_cast<const char*>(dw13s.ptr) + w13s_sz / 2,
                static_cast<const float*>(dw13a.ptr), static_cast<const float*>(dw13d.ptr),
                static_cast<const char*>(dw2c.ptr) + w2c_sz / 2,
                static_cast<const char*>(dw2s.ptr) + w2s_sz / 2,
                static_cast<const float*>(dw2a.ptr), static_cast<const float*>(dw2d.ptr), 1, 4.0f,
                25.0f, sc, tos1, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto gtp1 = to_f64_from_bf16(dout_tp1.download<std::uint16_t>(T * H));
            auto gs0  = to_f64_from_bf16(dout_s0.download<std::uint16_t>(T * H));
            auto gs1  = to_f64_from_bf16(dout_s1.download<std::uint16_t>(T * H));
            std::vector<double> gtp2(T * H);
            for (int i = 0; i < T * H; ++i) { gtp2[i] = gs0[i] + gs1[i]; }

            // FP64 independent oracle on token 0
            auto gtids = d_tids.download<std::int32_t>(T * 8);
            auto gtw   = d_tw.download<float>(T * 8);
            std::vector<double> ref0(H, 0.0);
            for (int kslot = 0; kslot < 8; ++kslot) {
                const int eid    = gtids[kslot];
                const double rw  = gtw[kslot];
                const double o13 = w13a[eid] / w13d[0];
                const double o2  = w2a[eid] / w2d[0];
                for (int slab = 0; slab < 2; ++slab) {
                    std::vector<double> act(I_slab);
                    for (int j = 0; j < I_slab; ++j) {
                        double gd = 0.0, ud = 0.0;
                        const std::size_t gr = (static_cast<std::size_t>(slab) * E + eid) * 1152 + 2 * j;
                        const std::size_t ur = gr + 1;
                        for (int g = 0; g < 144; ++g) {
                            double gsc = e4m3_to_f64(w13s[gr * 144 + g]);
                            double usc = e4m3_to_f64(w13s[ur * 144 + g]);
                            for (int b = 0; b < 8; ++b) {
                                std::uint8_t gb = w13c[gr * 1152 + g * 8 + b];
                                std::uint8_t ub = w13c[ur * 1152 + g * 8 + b];
                                double x0       = bf16_to_f32(hx[g * 16 + 2 * b]);
                                double x1       = bf16_to_f32(hx[g * 16 + 2 * b + 1]);
                                gd += (e2m1_to_f64(gb & 0xF) * x0 + e2m1_to_f64(gb >> 4) * x1) * gsc;
                                ud += (e2m1_to_f64(ub & 0xF) * x0 + e2m1_to_f64(ub >> 4) * x1) * usc;
                            }
                        }
                        gd *= o13;
                        ud *= o13;
                        double sig = 1.0 / (1.0 + std::exp(-gd));
                        double y   = (4.0 * std::tanh(gd / 4.0)) * sig * (25.0 * std::tanh(ud / 25.0)) * rw;
                        act[j]     = bf16_to_f32(f32_to_bf16(static_cast<float>(y)));
                    }
                    for (int row = 0; row < H; ++row) {
                        double acc          = 0.0;
                        const std::size_t r = (static_cast<std::size_t>(slab) * E + eid) * H + row;
                        for (int g = 0; g < 36; ++g) {
                            double wsc = e4m3_to_f64(w2s[r * 36 + g]);
                            for (int b = 0; b < 8; ++b) {
                                std::uint8_t wb = w2c[r * 288 + g * 8 + b];
                                acc += (e2m1_to_f64(wb & 0xF) * act[g * 16 + 2 * b] +
                                        e2m1_to_f64(wb >> 4) * act[g * 16 + 2 * b + 1]) * wsc;
                            }
                        }
                        ref0[row] += acc * o2;
                    }
                }
            }
            char lbl1[96], lbl2[96];
            std::snprintf(lbl1, sizeof(lbl1), "moe_nvfp4 T=%d TP1 vs FP64 oracle", T);
            std::snprintf(lbl2, sizeof(lbl2), "moe_nvfp4 T=%d TP1 == TP2_s0 + TP2_s1", T);
            check_close(lbl1, ref0, std::vector<double>(gtp1.begin(), gtp1.begin() + H), 0.15, 0.04);
            check_close(lbl2, gtp1, gtp2, 0.08, 0.02);
        }

        // Wave 3/4.5/5: router_fused(prep_mode=1) + skip_prep=true across T=1, 4, 8
        for (int T : {1, 4, 8}) {
            auto hx  = rand_bf16(rng, T * H, 0.3f);
            auto hrw = rand_bf16(rng, E * H, 0.05f);
            // Bias experts 0..11 strongly so multiple tokens collide on the same experts (mc=3, 4)!
            std::vector<float> hrb(E, -20.0f);
            for (int e = 0; e < 12; ++e) { hrb[e] = 2.0f + 0.05f * e; }
            DevBuf dx(hx.size() * 2), drw(hrw.size() * 2), dout_ref(T * H * 2), dout_fus(T * H * 2);
            dx.upload(hx); drw.upload(hrw); d_rbias.upload(hrb);
            Tensor tx = make_t(dx.ptr, DType::BF16, H, T), trw = make_t(drw.ptr, DType::BF16, H, E);
            Tensor trb = make_t(d_rbias.ptr, DType::FP32, E), trl = make_t(d_rlog.ptr, DType::FP32, E, T);
            Tensor tor = make_t(dout_ref.ptr, DType::BF16, H, T), tof = make_t(dout_fus.ptr, DType::BF16, H, T);

            ops::inkling2_bf16_linear(tx, trw, trl, stream);
            ops::inkling2_moe_route(trl, trb, 8.0f, sc, T, stream);
            ops::inkling2_moe_nvfp4_forward(tx, dw13c.ptr, dw13s.ptr,
                                            static_cast<const float*>(dw13a.ptr),
                                            static_cast<const float*>(dw13d.ptr), dw2c.ptr,
                                            dw2s.ptr, static_cast<const float*>(dw2a.ptr),
                                            static_cast<const float*>(dw2d.ptr), 1, 4.0f, 25.0f,
                                            sc, tor, stream, false, false);

            ops::inkling2_moe_router_fused(tx, trw, trb, 8.0f, sc, stream, 1);
            ops::inkling2_moe_nvfp4_forward(tx, dw13c.ptr, dw13s.ptr,
                                            static_cast<const float*>(dw13a.ptr),
                                            static_cast<const float*>(dw13d.ptr), dw2c.ptr,
                                            dw2s.ptr, static_cast<const float*>(dw2a.ptr),
                                            static_cast<const float*>(dw2d.ptr), 1, 4.0f, 25.0f,
                                            sc, tof, stream, true, false);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "moe_nvfp4 router_fused(prep=1)+skip_prep T=%d", T);
            if (T == 1) {
                if (dout_ref.download<std::uint16_t>(H) != dout_fus.download<std::uint16_t>(H)) {
                    throw std::runtime_error("moe_nvfp4 T=1 router_fused(prep=1)+PreStaged bit mismatch");
                }
                std::printf("  [PASS] %s (bit-exact)\n", lbl);
            } else {
                check_close(lbl, to_f64_from_bf16(dout_ref.download<std::uint16_t>(T * H)),
                            to_f64_from_bf16(dout_fus.download<std::uint16_t>(T * H)), 1e-2, 1e-2);
            }
        }
    }

    // --- MXFP8 MoE (T=1 and T=6, check TP1 vs FP64 AND TP1 == TP2_s0 + TP2_s1) ---
    {
        const std::size_t w13c_sz = 2ULL * E * 1152 * 2304;
        const std::size_t w13s_sz = 2ULL * E * 1152 * 72;
        const std::size_t w2c_sz  = 2ULL * E * H * 576;
        const std::size_t w2s_sz  = 2ULL * E * H * 18;
        std::vector<std::uint8_t> w13c(w13c_sz, 0), w13s(w13s_sz, 120);
        std::vector<std::uint8_t> w2c(w2c_sz, 0), w2s(w2s_sz, 120);
        for (int slab = 0; slab < 2; ++slab) {
            for (int e = 0; e < 12; ++e) {
                auto base13c = (static_cast<std::size_t>(slab) * E + e) * 1152 * 2304;
                auto base13s = (static_cast<std::size_t>(slab) * E + e) * 1152 * 72;
                auto c13     = rand_e4m3(1152 * 2304, 8);
                std::memcpy(&w13c[base13c], c13.data(), c13.size());
                for (std::size_t i = 0; i < 1152 * 72; ++i) {
                    w13s[base13s + i] = static_cast<std::uint8_t>(118 + (rng() % 6)); // 2^-9 .. 2^-4
                }
                auto base2c = (static_cast<std::size_t>(slab) * E + e) * H * 576;
                auto base2s = (static_cast<std::size_t>(slab) * E + e) * H * 18;
                auto c2     = rand_e4m3(static_cast<std::size_t>(H) * 576, 8);
                std::memcpy(&w2c[base2c], c2.data(), c2.size());
                for (std::size_t i = 0; i < static_cast<std::size_t>(H) * 18; ++i) {
                    w2s[base2s + i] = static_cast<std::uint8_t>(118 + (rng() % 6));
                }
            }
        }
        DevBuf dw13c(w13c_sz), dw13s(w13s_sz), dw2c(w2c_sz), dw2s(w2s_sz);
        dw13c.upload(w13c);
        dw13s.upload(w13s);
        dw2c.upload(w2c);
        dw2s.upload(w2s);

        for (int T : {1, 6}) {
            auto hx = rand_bf16(rng, T * H, 0.3f);
            std::vector<float> hrl(T * E, -10.0f), hrb(E, 0.0f);
            for (int t = 0; t < T; ++t) {
                for (int e = 0; e < 12; ++e) { hrl[t * E + e] = rand_f32(rng, 1, 2.0f)[0]; }
            }
            DevBuf dx(hx.size() * 2), dout_tp1(T * H * 2), dout_s0(T * H * 2), dout_s1(T * H * 2);
            dx.upload(hx);
            d_rlog.upload(hrl);
            d_rbias.upload(hrb);

            Tensor trl = make_t(d_rlog.ptr, DType::FP32, E, T);
            Tensor trb = make_t(d_rbias.ptr, DType::FP32, E);
            ops::inkling2_moe_route(trl, trb, 8.0f, sc, T, stream);

            Tensor tx   = make_t(dx.ptr, DType::BF16, H, T);
            Tensor to1  = make_t(dout_tp1.ptr, DType::BF16, H, T);
            Tensor tos0 = make_t(dout_s0.ptr, DType::BF16, H, T);
            Tensor tos1 = make_t(dout_s1.ptr, DType::BF16, H, T);

            ops::inkling2_moe_mxfp8_forward(tx, dw13c.ptr, dw13s.ptr, dw2c.ptr, dw2s.ptr, 2, 4.0f,
                                            25.0f, sc, to1, stream);
            ops::inkling2_moe_mxfp8_forward(tx, dw13c.ptr, dw13s.ptr, dw2c.ptr, dw2s.ptr, 1, 4.0f,
                                            25.0f, sc, tos0, stream);
            ops::inkling2_moe_mxfp8_forward(
                tx, static_cast<const char*>(dw13c.ptr) + w13c_sz / 2,
                static_cast<const char*>(dw13s.ptr) + w13s_sz / 2,
                static_cast<const char*>(dw2c.ptr) + w2c_sz / 2,
                static_cast<const char*>(dw2s.ptr) + w2s_sz / 2, 1, 4.0f, 25.0f, sc, tos1, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto gtp1 = to_f64_from_bf16(dout_tp1.download<std::uint16_t>(T * H));
            auto gs0  = to_f64_from_bf16(dout_s0.download<std::uint16_t>(T * H));
            auto gs1  = to_f64_from_bf16(dout_s1.download<std::uint16_t>(T * H));
            std::vector<double> gtp2(T * H);
            for (int i = 0; i < T * H; ++i) { gtp2[i] = gs0[i] + gs1[i]; }

            auto gtids = d_tids.download<std::int32_t>(T * 8);
            auto gtw   = d_tw.download<float>(T * 8);
            std::vector<double> ref0(H, 0.0);
            for (int kslot = 0; kslot < 8; ++kslot) {
                const int eid   = gtids[kslot];
                const double rw = gtw[kslot];
                for (int slab = 0; slab < 2; ++slab) {
                    std::vector<double> act(I_slab);
                    for (int j = 0; j < I_slab; ++j) {
                        double gd = 0.0, ud = 0.0;
                        const std::size_t gr = (static_cast<std::size_t>(slab) * E + eid) * 1152 + 2 * j;
                        const std::size_t ur = gr + 1;
                        for (int g = 0; g < 72; ++g) {
                            double gsc = ue8m0_to_f64(w13s[gr * 72 + g]);
                            double usc = ue8m0_to_f64(w13s[ur * 72 + g]);
                            for (int b = 0; b < 32; ++b) {
                                double xv = bf16_to_f32(hx[g * 32 + b]);
                                gd += e4m3_to_f64(w13c[gr * 2304 + g * 32 + b]) * xv * gsc;
                                ud += e4m3_to_f64(w13c[ur * 2304 + g * 32 + b]) * xv * usc;
                            }
                        }
                        double sig = 1.0 / (1.0 + std::exp(-gd));
                        double y   = (4.0 * std::tanh(gd / 4.0)) * sig * (25.0 * std::tanh(ud / 25.0)) * rw;
                        act[j]     = bf16_to_f32(f32_to_bf16(static_cast<float>(y)));
                    }
                    for (int row = 0; row < H; ++row) {
                        double acc          = 0.0;
                        const std::size_t r = (static_cast<std::size_t>(slab) * E + eid) * H + row;
                        for (int g = 0; g < 18; ++g) {
                            double wsc = ue8m0_to_f64(w2s[r * 18 + g]);
                            for (int b = 0; b < 32; ++b) {
                                acc += e4m3_to_f64(w2c[r * 576 + g * 32 + b]) * act[g * 32 + b] * wsc;
                            }
                        }
                        ref0[row] += acc;
                    }
                }
            }
            char lbl1[96], lbl2[96];
            std::snprintf(lbl1, sizeof(lbl1), "moe_mxfp8 T=%d TP1 vs FP64 oracle", T);
            std::snprintf(lbl2, sizeof(lbl2), "moe_mxfp8 T=%d TP1 == TP2_s0 + TP2_s1", T);
            check_close(lbl1, ref0, std::vector<double>(gtp1.begin(), gtp1.begin() + H), 0.10, 0.03);
            check_close(lbl2, gtp1, gtp2, 0.05, 0.02);
        }

        // Wave 3/4.5/5: router_fused(prep_mode=2) + skip_prep=true across T=1, 4, 8
        for (int T : {1, 4, 8}) {
            auto hx  = rand_bf16(rng, T * H, 0.3f);
            auto hrw = rand_bf16(rng, E * H, 0.05f);
            std::vector<float> hrb(E, -20.0f);
            for (int e = 0; e < 10; ++e) { hrb[e] = 2.0f + 0.05f * e; }
            DevBuf dx(hx.size() * 2), drw(hrw.size() * 2), dout_ref(T * H * 2), dout_fus(T * H * 2);
            dx.upload(hx); drw.upload(hrw); d_rbias.upload(hrb);
            Tensor tx = make_t(dx.ptr, DType::BF16, H, T), trw = make_t(drw.ptr, DType::BF16, H, E);
            Tensor trb = make_t(d_rbias.ptr, DType::FP32, E), trl = make_t(d_rlog.ptr, DType::FP32, E, T);
            Tensor tor = make_t(dout_ref.ptr, DType::BF16, H, T), tof = make_t(dout_fus.ptr, DType::BF16, H, T);

            ops::inkling2_bf16_linear(tx, trw, trl, stream);
            ops::inkling2_moe_route(trl, trb, 8.0f, sc, T, stream);
            ops::inkling2_moe_mxfp8_forward(tx, dw13c.ptr, dw13s.ptr, dw2c.ptr, dw2s.ptr, 1, 4.0f,
                                            25.0f, sc, tor, stream, false, false);

            ops::inkling2_moe_router_fused(tx, trw, trb, 8.0f, sc, stream, 2);
            ops::inkling2_moe_mxfp8_forward(tx, dw13c.ptr, dw13s.ptr, dw2c.ptr, dw2s.ptr, 1, 4.0f,
                                            25.0f, sc, tof, stream, true, false);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "moe_mxfp8 router_fused(prep=2)+skip_prep T=%d", T);
            if (T == 1) {
                if (dout_ref.download<std::uint16_t>(H) != dout_fus.download<std::uint16_t>(H)) {
                    throw std::runtime_error("moe_mxfp8 T=1 router_fused(prep=2)+PreStaged bit mismatch");
                }
                std::printf("  [PASS] %s (bit-exact)\n", lbl);
            } else {
                check_close(lbl, to_f64_from_bf16(dout_ref.download<std::uint16_t>(T * H)),
                            to_f64_from_bf16(dout_fus.download<std::uint16_t>(T * H)), 1e-2, 1e-2);
            }
        }
    }
}

// ============================================================================
// Test 6: DSpark MTP Ops
// ============================================================================
void test_dspark_ops(cudaStream_t stream) {
    std::mt19937 rng(505);
    const int H = 2304, T = 4, D = 128, n_q = 12, n_kv = 4;
    const float eps = 1e-6f;

    // mix_rmsnorm
    {
        auto ht  = rand_bf16(rng, T * 4 * H, 1.0f);
        auto hfw = rand_bf16(rng, 16, 1.0f);
        auto hnw = rand_bf16(rng, H, 1.0f);
        DevBuf dt(ht.size() * 2), dfw(32), dnw(H * 2), dout(T * H * 2);
        dt.upload(ht);
        dfw.upload(hfw);
        dnw.upload(hnw);
        Tensor tt  = make_t(dt.ptr, DType::BF16, H, 4, T);
        Tensor tfw = make_t(dfw.ptr, DType::BF16, 4, 4);
        Tensor tnw = make_t(dnw.ptr, DType::BF16, H);
        Tensor to  = make_t(dout.ptr, DType::BF16, H, T);
        ops::inkling2_mtp_mix_rmsnorm(tt, tfw, 2, tnw, eps, to, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        double fw[4], mx = -1e30, den = 0;
        for (int s = 0; s < 4; ++s) {
            fw[s] = bf16_to_f32(hfw[2 * 4 + s]);
            mx    = std::max(mx, fw[s]);
        }
        for (int s = 0; s < 4; ++s) {
            fw[s] = std::exp(fw[s] - mx);
            den += fw[s];
        }
        for (int s = 0; s < 4; ++s) { fw[s] /= den; }
        std::vector<double> ref(T * H);
        for (int t = 0; t < T; ++t) {
            double sq = 0;
            std::vector<double> m(H, 0.0);
            for (int i = 0; i < H; ++i) {
                for (int s = 0; s < 4; ++s) { m[i] += fw[s] * bf16_to_f32(ht[(t * 4 + s) * H + i]); }
                sq += m[i] * m[i];
            }
            double inv = 1.0 / std::sqrt(sq / H + eps);
            for (int i = 0; i < H; ++i) { ref[t * H + i] = m[i] * inv * bf16_to_f32(hnw[i]); }
        }
        check_close("mtp_mix_rmsnorm", ref, to_f64_from_bf16(dout.download<std::uint16_t>(T * H)),
                    2e-2, 2e-2);
    }

    // ctx_kv_write + block_qkv_rope + non-causal block_attention
    {
        const int ctx_len = 6, blk_sz = 8, ring_cap = 32;
        auto hkv_ctx = rand_bf16(rng, ctx_len * 2 * n_kv * D, 0.5f);
        auto hqkv    = rand_bf16(rng, blk_sz * (n_q + 2 * n_kv) * D, 0.5f);
        auto hqn     = rand_bf16(rng, D, 1.0f);
        auto hkn     = rand_bf16(rng, D, 1.0f);
        auto hknc    = rand_bf16(rng, D, 1.0f);

        std::vector<std::int32_t> hctx_pos(ctx_len), hblk_pos(blk_sz), hlanes_ctx(ctx_len, 0);
        for (int i = 0; i < ctx_len; ++i) { hctx_pos[i] = i; }
        for (int i = 0; i < blk_sz; ++i) { hblk_pos[i] = ctx_len + i; }
        std::vector<std::int32_t> hlane0 = {0}, hclen = {ctx_len};

        DevBuf dkvctx(hkv_ctx.size() * 2), dqkv(hqkv.size() * 2), dqn(D * 2), dkn(D * 2),
            dknc(D * 2), dctxpos(ctx_len * 4), dblkpos(blk_sz * 4), dlctx(ctx_len * 4),
            dl0(4), dclen(4), drk(ring_cap * n_kv * D * 2), drv(ring_cap * n_kv * D * 2),
            dq(blk_sz * n_q * D * 2), dkb(blk_sz * n_kv * D * 2), dvb(blk_sz * n_kv * D * 2),
            dout(blk_sz * n_q * D * 2);
        dkvctx.upload(hkv_ctx);
        dqkv.upload(hqkv);
        dqn.upload(hqn);
        dkn.upload(hkn);
        dknc.upload(hknc);
        dctxpos.upload(hctx_pos);
        dblkpos.upload(hblk_pos);
        dlctx.upload(hlanes_ctx);
        dl0.upload(hlane0);
        dclen.upload(hclen);

        ops::Inkling2KvTarget kv{};
        kv.is_local = true;
        kv.ring_k   = drk.ptr;
        kv.ring_v   = drv.ptr;
        kv.ring_cap = ring_cap;

        Tensor tkvc = make_t(dkvctx.ptr, DType::BF16, 2 * n_kv * D, ctx_len);
        Tensor tknc = make_t(dknc.ptr, DType::BF16, D);
        ops::inkling2_mtp_ctx_kv_write(tkvc, tknc, static_cast<const std::int32_t*>(dctxpos.ptr),
                                       static_cast<const std::int32_t*>(dlctx.ptr), ctx_len, n_kv,
                                       1e6f, eps, kv, stream);

        Tensor tqkv = make_t(dqkv.ptr, DType::BF16, (n_q + 2 * n_kv) * D, blk_sz);
        Tensor tqn  = make_t(dqn.ptr, DType::BF16, D);
        Tensor tkn  = make_t(dkn.ptr, DType::BF16, D);
        Tensor tq   = make_t(dq.ptr, DType::BF16, D, n_q, blk_sz);
        Tensor tkb  = make_t(dkb.ptr, DType::BF16, D, n_kv, blk_sz);
        Tensor tvb  = make_t(dvb.ptr, DType::BF16, D, n_kv, blk_sz);
        ops::inkling2_mtp_block_qkv_rope(tqkv, tqn, tkn,
                                         static_cast<const std::int32_t*>(dblkpos.ptr), blk_sz, n_q,
                                         n_kv, 1e6f, eps, tq, tkb, tvb, stream);

        Tensor to = make_t(dout.ptr, DType::BF16, n_q * D, blk_sz);
        ops::inkling2_mtp_block_attention(tq, tkb, tvb, kv,
                                          static_cast<const std::int32_t*>(dl0.ptr),
                                          static_cast<const std::int32_t*>(dclen.ptr), 1, blk_sz,
                                          n_q, n_kv, 1024, to, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto gq  = dq.download<std::uint16_t>(blk_sz * n_q * D);
        auto gkb = dkb.download<std::uint16_t>(blk_sz * n_kv * D);
        auto gvb = dvb.download<std::uint16_t>(blk_sz * n_kv * D);
        auto grk = drk.download<std::uint16_t>(ring_cap * n_kv * D);
        auto grv = drv.download<std::uint16_t>(ring_cap * n_kv * D);

        // FP64 check on non-causal attention
        std::vector<double> ref(blk_sz * n_q * D);
        const int tot_k = ctx_len + blk_sz;
        for (int bi = 0; bi < blk_sz; ++bi) {
            for (int qh = 0; qh < n_q; ++qh) {
                const int kvh = qh / 3;
                std::vector<double> scs(tot_k);
                double mx = -1e30;
                for (int ki = 0; ki < tot_k; ++ki) {
                    const auto* kp = (ki < ctx_len) ? &grk[(ki * n_kv + kvh) * D]
                                                    : &gkb[((ki - ctx_len) * n_kv + kvh) * D];
                    double dot     = 0.0;
                    for (int d = 0; d < D; ++d) {
                        dot += bf16_to_f32(gq[(bi * n_q + qh) * D + d]) * bf16_to_f32(kp[d]);
                    }
                    scs[ki] = dot / std::sqrt(static_cast<double>(D));
                    mx      = std::max(mx, scs[ki]);
                }
                double den = 0.0;
                std::vector<double> acc(D, 0.0);
                for (int ki = 0; ki < tot_k; ++ki) {
                    const auto* vp = (ki < ctx_len) ? &grv[(ki * n_kv + kvh) * D]
                                                    : &gvb[((ki - ctx_len) * n_kv + kvh) * D];
                    double p       = std::exp(scs[ki] - mx);
                    den += p;
                    for (int d = 0; d < D; ++d) { acc[d] += p * bf16_to_f32(vp[d]); }
                }
                for (int d = 0; d < D; ++d) { ref[(bi * n_q + qh) * D + d] = acc[d] / den; }
            }
        }
        check_close("mtp ctx_kv + qkv_rope + block_attn", ref,
                    to_f64_from_bf16(dout.download<std::uint16_t>(blk_sz * n_q * D)), 1.5e-2, 1.5e-2);
    }

    // silu_mul_packed + markov_rnn_prep + markov_rnn_gates + build_block_embed
    {
        const int inter = 512, B = 2, rank = 256, blk = 8;
        auto hgu = rand_bf16(rng, T * 2 * inter, 1.0f);
        DevBuf dgu(hgu.size() * 2), dsm(T * inter * 2);
        dgu.upload(hgu);
        Tensor tgu = make_t(dgu.ptr, DType::BF16, 2 * inter, T);
        Tensor tsm = make_t(dsm.ptr, DType::BF16, inter, T);
        ops::inkling2_silu_mul_packed(tgu, tsm, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> ref_sm(T * inter);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < inter; ++i) {
                double g = bf16_to_f32(hgu[t * 2 * inter + i]);
                double u = bf16_to_f32(hgu[t * 2 * inter + inter + i]);
                ref_sm[t * inter + i] = (g / (1.0 + std::exp(-g))) * u;
            }
        }
        check_close("silu_mul_packed", ref_sm,
                    to_f64_from_bf16(dsm.download<std::uint16_t>(T * inter)), 1.5e-2, 1.5e-2);

        auto hst  = rand_bf16(rng, B * rank, 0.5f);
        auto hw1  = rand_bf16(rng, 100 * rank, 0.5f);
        auto hhb  = rand_bf16(rng, B * blk * H, 0.5f);
        auto hfr  = rand_bf16(rng, B * H, 0.5f);
        auto hpr  = rand_bf16(rng, B * 3 * rank, 0.8f);
        std::vector<std::int32_t> hprev = {7, 42};
        const int width = 2 * rank + 2 * H;

        DevBuf dst(hst.size() * 2), dw1(hw1.size() * 2), dhb(hhb.size() * 2), dfr(hfr.size() * 2),
            dprev(B * 4), dz(B * width * 2), dpr(hpr.size() * 2), dlat(B * rank * 2);
        dst.upload(hst);
        dw1.upload(hw1);
        dhb.upload(hhb);
        dfr.upload(hfr);
        dprev.upload(hprev);
        dpr.upload(hpr);

        Tensor tst   = make_t(dst.ptr, DType::BF16, rank, B);
        Tensor tprev = make_t(dprev.ptr, DType::I32, B);
        Tensor tw1   = make_t(dw1.ptr, DType::BF16, rank, 100);
        Tensor thb   = make_t(dhb.ptr, DType::BF16, H, B * blk);
        Tensor tfr   = make_t(dfr.ptr, DType::BF16, H, B);
        Tensor tz    = make_t(dz.ptr, DType::BF16, width, B);
        ops::inkling2_mtp_markov_rnn_prep(tst, tprev, tw1, thb, 3, blk, tfr, tz, stream);

        Tensor tpr  = make_t(dpr.ptr, DType::BF16, 3 * rank, B);
        Tensor tlat = make_t(dlat.ptr, DType::BF16, rank, B);
        ops::inkling2_mtp_markov_rnn_gates(tpr, tst, tlat, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto gz = dz.download<std::uint16_t>(B * width);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < rank; ++i) {
                if (gz[b * width + i] != hst[b * rank + i]) { throw std::runtime_error("rnn_prep state"); }
                if (gz[b * width + rank + i] != hw1[hprev[b] * rank + i]) {
                    throw std::runtime_error("rnn_prep w1");
                }
            }
            for (int i = 0; i < H; ++i) {
                if (gz[b * width + 2 * rank + i] != hhb[(b * blk + 3) * H + i]) {
                    throw std::runtime_error("rnn_prep hidden");
                }
                if (gz[b * width + 2 * rank + H + i] != hfr[b * H + i]) {
                    throw std::runtime_error("rnn_prep f_rnn");
                }
            }
        }
        std::vector<double> ref_st(B * rank), ref_lat(B * rank);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < rank; ++i) {
                double g = 1.0 / (1.0 + std::exp(-bf16_to_f32(hpr[b * 3 * rank + i])));
                double c = std::tanh(bf16_to_f32(hpr[b * 3 * rank + rank + i]));
                double o = std::tanh(bf16_to_f32(hpr[b * 3 * rank + 2 * rank + i]));
                ref_st[b * rank + i]  = g * bf16_to_f32(hst[b * rank + i]) + (1.0 - g) * c;
                ref_lat[b * rank + i] = o;
            }
        }
        check_close("markov_rnn_gates state", ref_st,
                    to_f64_from_bf16(dst.download<std::uint16_t>(B * rank)), 1.5e-2, 1.5e-2);
        check_close("markov_rnn_gates latent", ref_lat,
                    to_f64_from_bf16(dlat.download<std::uint16_t>(B * rank)), 1.5e-2, 1.5e-2);
    }

    // Indexed mtp_mix_rmsnorm (H=2304 vectorized path + non-trivial src/dst scatter-gather)
    {
        const int H2 = 2304, cap_T = 10, count = 4;
        auto hcaps = rand_bf16(rng, cap_T * 4 * H2, 0.7f);
        auto hfw   = rand_bf16(rng, 4 * 4, 0.5f);
        auto hnw   = rand_bf16(rng, H2, 1.0f);
        std::vector<std::int32_t> hsrc = {7, 2, 9, 0};
        std::vector<std::int32_t> hdst = {3, 1, 0, 2};
        DevBuf dcaps(hcaps.size() * 2), dfw(hfw.size() * 2), dnw(hnw.size() * 2),
            dsrc(count * 4), ddst(count * 4), dout(count * H2 * 2);
        dcaps.upload(hcaps); dfw.upload(hfw); dnw.upload(hnw);
        dsrc.upload(hsrc); ddst.upload(hdst);
        dout.zero();
        Tensor tcaps = make_t(dcaps.ptr, DType::BF16, H2, 4, cap_T);
        Tensor tfw   = make_t(dfw.ptr, DType::BF16, 4, 4);
        Tensor tnw   = make_t(dnw.ptr, DType::BF16, H2);
        Tensor tout  = make_t(dout.ptr, DType::BF16, H2, count);
        ops::inkling2_mtp_mix_rmsnorm_indexed(
            tcaps, static_cast<const std::int32_t*>(dsrc.ptr),
            static_cast<const std::int32_t*>(ddst.ptr), count, tfw, 2, tnw, 1e-6f, tout, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        double fw[4], mx = -1e30, den = 0.0;
        for (int s = 0; s < 4; ++s) { fw[s] = bf16_to_f32(hfw[2 * 4 + s]); mx = std::max(mx, fw[s]); }
        for (int s = 0; s < 4; ++s) { fw[s] = std::exp(fw[s] - mx); den += fw[s]; }
        for (int s = 0; s < 4; ++s) { fw[s] /= den; }
        std::vector<double> ref(count * H2, 0.0);
        for (int t = 0; t < count; ++t) {
            const int st = hsrc[t], dt = hdst[t];
            double sq = 0.0;
            std::vector<double> v(H2, 0.0);
            for (int i = 0; i < H2; ++i) {
                for (int s = 0; s < 4; ++s) { v[i] += fw[s] * bf16_to_f32(hcaps[(st * 4 + s) * H2 + i]); }
                sq += v[i] * v[i];
            }
            double inv = 1.0 / std::sqrt(sq / H2 + 1e-6);
            for (int i = 0; i < H2; ++i) { ref[dt * H2 + i] = v[i] * inv * bf16_to_f32(hnw[i]); }
        }
        check_close("mtp_mix_rmsnorm_indexed (H=2304)", ref,
                    to_f64_from_bf16(dout.download<std::uint16_t>(count * H2)), 2e-2, 1.5e-2);
    }

    // Wave 4: mtp_mix_rmsnorm_batch3_and_frnn vs 4 separate mtp_mix_rmsnorm_indexed calls
    {
        const int cap_T = 10, ctx_cnt = 5, frnn_cnt = 2;
        auto hcaps = rand_bf16(rng, cap_T * 4 * H, 0.6f);
        auto hfw   = rand_bf16(rng, 16, 0.5f);
        auto hn0   = rand_bf16(rng, H, 1.0f), hn1 = rand_bf16(rng, H, 1.0f),
             hn2   = rand_bf16(rng, H, 1.0f), hnf = rand_bf16(rng, H, 1.0f);
        std::vector<std::int32_t> hcsrc = {8, 1, 5, 3, 9}, hfsrc = {4, 2}, hfdst = {1, 0};
        DevBuf dcaps(hcaps.size() * 2), dfw(32), dn0(H * 2), dn1(H * 2), dn2(H * 2), dnf(H * 2);
        DevBuf dcsrc(ctx_cnt * 4), dfsrc(frnn_cnt * 4), dfdst(frnn_cnt * 4);
        DevBuf dor0(ctx_cnt * H * 2), dor1(ctx_cnt * H * 2), dor2(ctx_cnt * H * 2), dorf(frnn_cnt * H * 2);
        DevBuf dof0(ctx_cnt * H * 2), dof1(ctx_cnt * H * 2), dof2(ctx_cnt * H * 2), doff(frnn_cnt * H * 2);
        dcaps.upload(hcaps); dfw.upload(hfw);
        dn0.upload(hn0); dn1.upload(hn1); dn2.upload(hn2); dnf.upload(hnf);
        dcsrc.upload(hcsrc); dfsrc.upload(hfsrc); dfdst.upload(hfdst);
        Tensor tcaps = make_t(dcaps.ptr, DType::BF16, H, 4, cap_T), tfw = make_t(dfw.ptr, DType::BF16, 4, 4);
        Tensor tn0 = make_t(dn0.ptr, DType::BF16, H), tn1 = make_t(dn1.ptr, DType::BF16, H),
               tn2 = make_t(dn2.ptr, DType::BF16, H), tnf = make_t(dnf.ptr, DType::BF16, H);
        Tensor tor0 = make_t(dor0.ptr, DType::BF16, H, ctx_cnt), tor1 = make_t(dor1.ptr, DType::BF16, H, ctx_cnt),
               tor2 = make_t(dor2.ptr, DType::BF16, H, ctx_cnt), torf = make_t(dorf.ptr, DType::BF16, H, frnn_cnt);
        Tensor tof0 = make_t(dof0.ptr, DType::BF16, H, ctx_cnt), tof1 = make_t(dof1.ptr, DType::BF16, H, ctx_cnt),
               tof2 = make_t(dof2.ptr, DType::BF16, H, ctx_cnt), toff = make_t(doff.ptr, DType::BF16, H, frnn_cnt);
        const auto* cs = static_cast<const std::int32_t*>(dcsrc.ptr);
        const auto* fs = static_cast<const std::int32_t*>(dfsrc.ptr);
        const auto* fd = static_cast<const std::int32_t*>(dfdst.ptr);

        ops::inkling2_mtp_mix_rmsnorm_indexed(tcaps, cs, nullptr, ctx_cnt, tfw, 0, tn0, eps, tor0, stream);
        ops::inkling2_mtp_mix_rmsnorm_indexed(tcaps, cs, nullptr, ctx_cnt, tfw, 1, tn1, eps, tor1, stream);
        ops::inkling2_mtp_mix_rmsnorm_indexed(tcaps, cs, nullptr, ctx_cnt, tfw, 2, tn2, eps, tor2, stream);
        ops::inkling2_mtp_mix_rmsnorm_indexed(tcaps, fs, fd, frnn_cnt, tfw, 3, tnf, eps, torf, stream);
        ops::inkling2_mtp_mix_rmsnorm_batch3_and_frnn(tcaps, cs, ctx_cnt, fs, fd, frnn_cnt, tfw,
                                                      tn0, tn1, tn2, tnf, eps, tof0, tof1, tof2,
                                                      toff, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close("mtp_mix_batch3_and_frnn out0 exact", to_f64_from_bf16(dor0.download<std::uint16_t>(ctx_cnt * H)), to_f64_from_bf16(dof0.download<std::uint16_t>(ctx_cnt * H)), 0.0, 0.0);
        check_close("mtp_mix_batch3_and_frnn out1 exact", to_f64_from_bf16(dor1.download<std::uint16_t>(ctx_cnt * H)), to_f64_from_bf16(dof1.download<std::uint16_t>(ctx_cnt * H)), 0.0, 0.0);
        check_close("mtp_mix_batch3_and_frnn out2 exact", to_f64_from_bf16(dor2.download<std::uint16_t>(ctx_cnt * H)), to_f64_from_bf16(dof2.download<std::uint16_t>(ctx_cnt * H)), 0.0, 0.0);
        check_close("mtp_mix_batch3_and_frnn frnn exact", to_f64_from_bf16(dorf.download<std::uint16_t>(frnn_cnt * H)), to_f64_from_bf16(doff.download<std::uint16_t>(frnn_cnt * H)), 0.0, 0.0);
    }

    // Wave 4: mtp_ctx_kv_write_3 (ring + paged + ring) vs 3 separate calls
    {
        const int tok = 5, ring_cap = 32, max_pages = 4;
        auto hkv0 = rand_bf16(rng, tok * 2 * n_kv * D, 0.5f),
             hkv1 = rand_bf16(rng, tok * 2 * n_kv * D, 0.5f),
             hkv2 = rand_bf16(rng, tok * 2 * n_kv * D, 0.5f);
        auto hkn0 = rand_bf16(rng, D, 1.0f), hkn1 = rand_bf16(rng, D, 1.0f), hkn2 = rand_bf16(rng, D, 1.0f);
        std::vector<std::int32_t> hpos = {2, 3, 4, 64, 65}, hlanes = {0, 0, 0, 0, 0}, hbt = {3, 1, 0, 2};
        DevBuf dkv0(hkv0.size() * 2), dkv1(hkv1.size() * 2), dkv2(hkv2.size() * 2);
        DevBuf dkn0(D * 2), dkn1(D * 2), dkn2(D * 2), dpos(tok * 4), dl(tok * 4), dbt(max_pages * 4);
        dkv0.upload(hkv0); dkv1.upload(hkv1); dkv2.upload(hkv2);
        dkn0.upload(hkn0); dkn1.upload(hkn1); dkn2.upload(hkn2);
        dpos.upload(hpos); dl.upload(hlanes); dbt.upload(hbt);
        DevBuf rk0r(ring_cap * n_kv * D * 2), rv0r(ring_cap * n_kv * D * 2),
               pk1r(max_pages * n_kv * 64 * D * 2), pv1r(max_pages * n_kv * 64 * D * 2),
               rk2r(ring_cap * n_kv * D * 2), rv2r(ring_cap * n_kv * D * 2);
        DevBuf rk0f(ring_cap * n_kv * D * 2), rv0f(ring_cap * n_kv * D * 2),
               pk1f(max_pages * n_kv * 64 * D * 2), pv1f(max_pages * n_kv * 64 * D * 2),
               rk2f(ring_cap * n_kv * D * 2), rv2f(ring_cap * n_kv * D * 2);
        rk0r.zero(); rv0r.zero(); pk1r.zero(); pv1r.zero(); rk2r.zero(); rv2r.zero();
        rk0f.zero(); rv0f.zero(); pk1f.zero(); pv1f.zero(); rk2f.zero(); rv2f.zero();

        ops::Inkling2KvTarget kv0r{}, kv1r{}, kv2r{}, kv0f{}, kv1f{}, kv2f{};
        kv0r.is_local = true;  kv0r.ring_k = rk0r.ptr; kv0r.ring_v = rv0r.ptr; kv0r.ring_cap = ring_cap;
        kv1r.is_local = false; kv1r.paged_k = pk1r.ptr; kv1r.paged_v = pv1r.ptr; kv1r.block_tables = static_cast<const std::int32_t*>(dbt.ptr); kv1r.max_logical_pages = max_pages;
        kv2r.is_local = true;  kv2r.ring_k = rk2r.ptr; kv2r.ring_v = rv2r.ptr; kv2r.ring_cap = ring_cap;
        kv0f.is_local = true;  kv0f.ring_k = rk0f.ptr; kv0f.ring_v = rv0f.ptr; kv0f.ring_cap = ring_cap;
        kv1f.is_local = false; kv1f.paged_k = pk1f.ptr; kv1f.paged_v = pv1f.ptr; kv1f.block_tables = static_cast<const std::int32_t*>(dbt.ptr); kv1f.max_logical_pages = max_pages;
        kv2f.is_local = true;  kv2f.ring_k = rk2f.ptr; kv2f.ring_v = rv2f.ptr; kv2f.ring_cap = ring_cap;

        Tensor tk0 = make_t(dkv0.ptr, DType::BF16, 2 * n_kv * D, tok),
               tk1 = make_t(dkv1.ptr, DType::BF16, 2 * n_kv * D, tok),
               tk2 = make_t(dkv2.ptr, DType::BF16, 2 * n_kv * D, tok);
        Tensor tn0 = make_t(dkn0.ptr, DType::BF16, D), tn1 = make_t(dkn1.ptr, DType::BF16, D),
               tn2 = make_t(dkn2.ptr, DType::BF16, D);
        const auto* pp = static_cast<const std::int32_t*>(dpos.ptr);
        const auto* lp = static_cast<const std::int32_t*>(dl.ptr);
        ops::inkling2_mtp_ctx_kv_write(tk0, tn0, pp, lp, tok, n_kv, 1e6f, eps, kv0r, stream);
        ops::inkling2_mtp_ctx_kv_write(tk1, tn1, pp, lp, tok, n_kv, 1e6f, eps, kv1r, stream);
        ops::inkling2_mtp_ctx_kv_write(tk2, tn2, pp, lp, tok, n_kv, 1e6f, eps, kv2r, stream);
        ops::inkling2_mtp_ctx_kv_write_3(tk0, tn0, kv0f, tk1, tn1, kv1f, tk2, tn2, kv2f,
                                         pp, lp, tok, n_kv, 1e6f, eps, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close("mtp_ctx_kv_write_3 L0 K exact", to_f64_from_bf16(rk0r.download<std::uint16_t>(ring_cap * n_kv * D)), to_f64_from_bf16(rk0f.download<std::uint16_t>(ring_cap * n_kv * D)), 0.0, 0.0);
        check_close("mtp_ctx_kv_write_3 L1 K exact", to_f64_from_bf16(pk1r.download<std::uint16_t>(max_pages * n_kv * 64 * D)), to_f64_from_bf16(pk1f.download<std::uint16_t>(max_pages * n_kv * 64 * D)), 0.0, 0.0);
        check_close("mtp_ctx_kv_write_3 L2 V exact", to_f64_from_bf16(rv2r.download<std::uint16_t>(ring_cap * n_kv * D)), to_f64_from_bf16(rv2f.download<std::uint16_t>(ring_cap * n_kv * D)), 0.0, 0.0);
    }

    // Wave 4: mtp_block_attention long-context adaptive Last-CTA Split-KV (ctx_len=1100, SWA & Paged Global)
    // ctx_len = 4100 (global only) selects the Wave 5 single-exp online softmax route (>= 160 keys/split).
    for (const int ctx_len : {1100, 4100}) for (bool is_local : {true, false}) {
        if (is_local && ctx_len != 1100) { continue; }
        const int blk_sz = 4, swa_win = 1024, ring_cap = 1024, max_pages = (ctx_len + 63) / 64 + 4;
        auto hq  = rand_bf16(rng, blk_sz * n_q * D, 0.4f);
        auto hkb = rand_bf16(rng, blk_sz * n_kv * D, 0.4f);
        auto hvb = rand_bf16(rng, blk_sz * n_kv * D, 0.4f);
        auto hkc = rand_bf16(rng, ctx_len * n_kv * D, 0.35f);
        auto hvc = rand_bf16(rng, ctx_len * n_kv * D, 0.35f);
        std::vector<std::int32_t> hbt(max_pages);
        // Deterministic physical-page permutation (independent of the data RNG stream).
        std::iota(hbt.begin(), hbt.end(), 0);
        std::shuffle(hbt.begin(), hbt.end(), std::mt19937(911));
        std::vector<std::uint16_t> hrk(ring_cap * n_kv * D, 0), hrv(ring_cap * n_kv * D, 0);
        std::vector<std::uint16_t> hpk(max_pages * n_kv * 64 * D, 0), hpv(max_pages * n_kv * 64 * D, 0);
        for (int kp = 0; kp < ctx_len; ++kp) {
            const int slot = kp % ring_cap, pp = hbt[kp >> 6], ps = kp & 63;
            for (int h = 0; h < n_kv; ++h) {
                for (int d = 0; d < D; ++d) {
                    hrk[(slot * n_kv + h) * D + d]           = hkc[(kp * n_kv + h) * D + d];
                    hrv[(slot * n_kv + h) * D + d]           = hvc[(kp * n_kv + h) * D + d];
                    hpk[((pp * n_kv + h) * 64 + ps) * D + d] = hkc[(kp * n_kv + h) * D + d];
                    hpv[((pp * n_kv + h) * 64 + ps) * D + d] = hvc[(kp * n_kv + h) * D + d];
                }
            }
        }
        std::vector<std::int32_t> hl0 = {0}, hcl = {ctx_len};
        DevBuf dq(hq.size() * 2), dkb(hkb.size() * 2), dvb(hvb.size() * 2);
        DevBuf drk(hrk.size() * 2), drv(hrv.size() * 2), dpk(hpk.size() * 2), dpv(hpv.size() * 2);
        DevBuf dbt(max_pages * 4), dl0(4), dcl(4), dout(blk_sz * n_q * D * 2);
        DevBuf dsplit(blk_sz * n_q * 16 * (D + 4) * 4), dsdone(blk_sz * n_q * 4);
        dq.upload(hq); dkb.upload(hkb); dvb.upload(hvb);
        drk.upload(hrk); drv.upload(hrv); dpk.upload(hpk); dpv.upload(hpv);
        dbt.upload(hbt); dl0.upload(hl0); dcl.upload(hcl); dsdone.zero();

        ops::Inkling2KvTarget kv{};
        kv.is_local = is_local; kv.ring_k = drk.ptr; kv.ring_v = drv.ptr; kv.ring_cap = ring_cap;
        kv.paged_k = dpk.ptr; kv.paged_v = dpv.ptr;
        kv.block_tables = static_cast<const std::int32_t*>(dbt.ptr); kv.max_logical_pages = max_pages;
        Tensor tq = make_t(dq.ptr, DType::BF16, D, n_q, blk_sz);
        Tensor tkb = make_t(dkb.ptr, DType::BF16, D, n_kv, blk_sz);
        Tensor tvb = make_t(dvb.ptr, DType::BF16, D, n_kv, blk_sz);
        Tensor to  = make_t(dout.ptr, DType::BF16, n_q * D, blk_sz);
        ops::inkling2_mtp_block_attention(tq, tkb, tvb, kv, static_cast<const std::int32_t*>(dl0.ptr),
                                          static_cast<const std::int32_t*>(dcl.ptr), 1, blk_sz,
                                          n_q, n_kv, swa_win, to, stream,
                                          static_cast<float*>(dsplit.ptr), 16,
                                          static_cast<std::int32_t*>(dsdone.ptr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int v : dsdone.download<std::int32_t>(blk_sz * n_q)) {
            if (v != 0) { throw std::runtime_error("mtp_block_attention split_done not reset to 0"); }
        }
        // Independent FP64 oracle on block pos 2
        const int bi = 2;
        const int ctx_start = is_local ? std::max(0, ctx_len - swa_win) : 0;
        const int ctx_k     = ctx_len - ctx_start;
        const int tot_k     = ctx_k + blk_sz;
        std::vector<double> ref(n_q * D);
        for (int qh = 0; qh < n_q; ++qh) {
            const int kvh = qh / 3;
            std::vector<double> scs(tot_k);
            double mx = -1e30;
            for (int ki = 0; ki < tot_k; ++ki) {
                const auto* kp = (ki < ctx_k) ? &hkc[((ctx_start + ki) * n_kv + kvh) * D]
                                              : &hkb[((ki - ctx_k) * n_kv + kvh) * D];
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += bf16_to_f32(hq[(bi * n_q + qh) * D + d]) * bf16_to_f32(kp[d]);
                }
                scs[ki] = dot / std::sqrt(static_cast<double>(D));
                mx      = std::max(mx, scs[ki]);
            }
            double den = 0.0;
            std::vector<double> acc(D, 0.0);
            for (int ki = 0; ki < tot_k; ++ki) {
                const auto* vp = (ki < ctx_k) ? &hvc[((ctx_start + ki) * n_kv + kvh) * D]
                                              : &hvb[((ki - ctx_k) * n_kv + kvh) * D];
                double p = std::exp(scs[ki] - mx);
                den += p;
                for (int d = 0; d < D; ++d) { acc[d] += p * bf16_to_f32(vp[d]); }
            }
            for (int d = 0; d < D; ++d) { ref[qh * D + d] = acc[d] / den; }
        }
        auto got_all = to_f64_from_bf16(dout.download<std::uint16_t>(blk_sz * n_q * D));
        char mlbl[96];
        std::snprintf(mlbl, sizeof(mlbl), "mtp_block_attn %s long-ctx Split-KV (%d) vs FP64",
                      is_local ? "SWA" : "Full", ctx_len);
        check_close(mlbl,
                    ref, std::vector<double>(got_all.begin() + bi * n_q * D, got_all.begin() + (bi + 1) * n_q * D),
                    1.5e-2, 1.5e-2);
    }

    // Wave 4: mtp_final_norm_extract_zero + fused mtp_markov_rnn_step (2 ping-pong steps vs unfused!)
    {
        const int B = 2, blk = 4, rank = 256, width = 2 * rank + 2 * H;
        auto hhb = rand_bf16(rng, B * blk * H, 0.7f);
        auto hnw = rand_bf16(rng, H, 1.0f);
        auto hst = rand_bf16(rng, B * rank, 0.6f);
        DevBuf dhb_r(hhb.size() * 2), dhb_f(hhb.size() * 2), dnw(H * 2);
        DevBuf ddr_r(B * (blk - 1) * H * 2), ddr_f(B * (blk - 1) * H * 2), dst_z(B * rank * 2);
        dhb_r.upload(hhb); dhb_f.upload(hhb); dnw.upload(hnw); dst_z.upload(hst);
        Tensor thb_r = make_t(dhb_r.ptr, DType::BF16, H, B * blk);
        Tensor thb_f = make_t(dhb_f.ptr, DType::BF16, H, B * blk);
        Tensor tnw   = make_t(dnw.ptr, DType::BF16, H);
        Tensor tdr_r = make_t(ddr_r.ptr, DType::BF16, H, B * (blk - 1));
        Tensor tdr_f = make_t(ddr_f.ptr, DType::BF16, H, B * (blk - 1));
        Tensor tst_z = make_t(dst_z.ptr, DType::BF16, rank, B);

        ops::inkling2_rmsnorm(thb_r, &tnw, thb_r, eps, stream);
        ops::inkling2_mtp_extract_draft_hiddens(thb_r, B, blk, tdr_r, stream);
        ops::inkling2_mtp_final_norm_extract_zero(thb_f, tnw, eps, B, blk, tdr_f, tst_z, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close("mtp_final_norm_extract_zero hidden", to_f64_from_bf16(dhb_r.download<std::uint16_t>(hhb.size())), to_f64_from_bf16(dhb_f.download<std::uint16_t>(hhb.size())), 1e-2, 1e-2);
        check_close("mtp_final_norm_extract_zero draft", to_f64_from_bf16(ddr_r.download<std::uint16_t>(B * (blk - 1) * H)), to_f64_from_bf16(ddr_f.download<std::uint16_t>(B * (blk - 1) * H)), 1e-2, 1e-2);
        for (std::uint16_t z : dst_z.download<std::uint16_t>(B * rank)) {
            if (z != 0) { throw std::runtime_error("mtp_final_norm_extract_zero did not zero rnn_state"); }
        }

        // Fused Markov RNN step (2 sequential ping-pong steps) vs prep + bf16_linear + gates
        auto hw1 = rand_bf16(rng, 64 * rank, 0.4f);
        auto hfr = rand_bf16(rng, B * H, 0.4f);
        auto hpw = rand_bf16(rng, 3 * rank * width, 0.05f);
        auto hpb = rand_bf16(rng, 3 * rank, 0.2f);
        std::vector<std::int32_t> hp0 = {5, 19}, hp1 = {33, 2};
        DevBuf dw1(hw1.size() * 2), dfr(hfr.size() * 2), dpw(hpw.size() * 2), dpb(hpb.size() * 2);
        DevBuf dp0(B * 4), dp1(B * 4);
        DevBuf dst_ref(B * rank * 2), dst_pp0(B * rank * 2), dst_pp1(B * rank * 2);
        DevBuf dz(B * width * 2), dpr(B * 3 * rank * 2), dlat_r(B * rank * 2), dlat_f(B * rank * 2);
        dw1.upload(hw1); dfr.upload(hfr); dpw.upload(hpw); dpb.upload(hpb);
        dp0.upload(hp0); dp1.upload(hp1);
        dst_ref.upload(hst); dst_pp0.upload(hst);

        Tensor tw1   = make_t(dw1.ptr, DType::BF16, rank, 64);
        Tensor tfr   = make_t(dfr.ptr, DType::BF16, H, B);
        Tensor tpw   = make_t(dpw.ptr, DType::BF16, width, 3 * rank);
        Tensor tpb   = make_t(dpb.ptr, DType::BF16, 3 * rank);
        Tensor tp0   = make_t(dp0.ptr, DType::I32, B), tp1 = make_t(dp1.ptr, DType::I32, B);
        Tensor tst_r = make_t(dst_ref.ptr, DType::BF16, rank, B);
        Tensor tst_0 = make_t(dst_pp0.ptr, DType::BF16, rank, B), tst_1 = make_t(dst_pp1.ptr, DType::BF16, rank, B);
        Tensor tz    = make_t(dz.ptr, DType::BF16, width, B), tpr = make_t(dpr.ptr, DType::BF16, 3 * rank, B);
        Tensor tlr   = make_t(dlat_r.ptr, DType::BF16, rank, B), tlf = make_t(dlat_f.ptr, DType::BF16, rank, B);

        // Step 0 (block_pos = 1): pp0 -> pp1
        ops::inkling2_mtp_markov_rnn_prep(tst_r, tp0, tw1, thb_r, 1, blk, tfr, tz, stream);
        ops::inkling2_bf16_linear(tz, tpw, tpr, stream, &tpb);
        ops::inkling2_mtp_markov_rnn_gates(tpr, tst_r, tlr, stream);
        ops::inkling2_mtp_markov_rnn_step(tst_0, tp0, tw1, thb_r, 1, blk, tfr, tpw, tpb, tst_1, tlf, stream);

        // Step 1 (block_pos = 2): pp1 -> pp0
        ops::inkling2_mtp_markov_rnn_prep(tst_r, tp1, tw1, thb_r, 2, blk, tfr, tz, stream);
        ops::inkling2_bf16_linear(tz, tpw, tpr, stream, &tpb);
        ops::inkling2_mtp_markov_rnn_gates(tpr, tst_r, tlr, stream);
        ops::inkling2_mtp_markov_rnn_step(tst_1, tp1, tw1, thb_r, 2, blk, tfr, tpw, tpb, tst_0, tlf, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_close("mtp_markov_rnn_step (2-step ping-pong) state", to_f64_from_bf16(dst_ref.download<std::uint16_t>(B * rank)), to_f64_from_bf16(dst_pp0.download<std::uint16_t>(B * rank)), 2e-2, 2e-2);
        check_close("mtp_markov_rnn_step (2-step ping-pong) latent", to_f64_from_bf16(dlat_r.download<std::uint16_t>(B * rank)), to_f64_from_bf16(dlat_f.download<std::uint16_t>(B * rank)), 2e-2, 2e-2);
    }
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::printf("No CUDA device; skipping\n");
        return 77;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    try {
        std::printf("=== Testing Inkling-2-Mini CUDA Ops ===\n");
        test_bf16_linear_and_heads(stream);
        test_norms_attnres_controller(stream);
        test_sconv(stream);
        test_attention(stream);
        test_sparse_moe(stream);
        test_dspark_ops(stream);
        std::printf("=== ALL INKLING-2-MINI OP TESTS PASSED ===\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILURE: %s\n", e.what());
        (void)cudaStreamDestroy(stream);
        return 1;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
