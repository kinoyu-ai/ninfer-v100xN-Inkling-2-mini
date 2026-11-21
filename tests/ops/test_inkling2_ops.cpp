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
    }
    template <typename T>
    std::vector<T> download(std::size_t count) const {
        std::vector<T> v(count);
        CUDA_CHECK(cudaMemcpy(v.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void zero() {
        if (bytes > 0) { CUDA_CHECK(cudaMemset(ptr, 0, bytes)); }
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
        DevBuf dsplit(S_dec * n_q * 16 * (D + 2) * 4);
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
