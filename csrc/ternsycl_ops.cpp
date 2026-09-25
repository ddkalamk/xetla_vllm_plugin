// Torch ops for the TernSYCL kernels: torch.ops.ternsycl.*. Same signatures,
// layouts and dtypes as the ternsycl ops they replace. The registration sits in a
// .cpp: global ctors of .sycl TUs are not always run at .so load time.

#include <ATen/record_function.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <cstdint>
#include <optional>

sycl::event ternsycl_upcvt_run(sycl::queue &q, bool bf16, const void *A, const int32_t *B, const void *S, void *C,
        const void *other, int postop, int M, int N, int K);
sycl::event ternsycl_int8_run(sycl::queue &q, bool bf16, const void *A, const int32_t *B, const void *SB, void *C,
        void *SA, int8_t *Aq, int M, int N, int K);
int ternsycl_int8_ldsa(int M);
sycl::event ternsycl_hadamard_run(sycl::queue &q, bool bf16, const void *x, const int8_t *signs, void *y,
        int64_t rows, int K, bool inverse);

namespace {

sycl::queue &queue_of(const torch::Tensor &t) { return c10::xpu::getCurrentXPUStream(t.device().index()).queue(); }

// A [M, K] DT, B [K/16, N] int32, S [K/128, N] DT; N % 16, K % 128.
void check_gemm(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &S, at::ScalarType dt) {
    TORCH_CHECK(A.scalar_type() == dt && S.scalar_type() == dt, "A and scale_B must be ", dt);
    TORCH_CHECK(B.scalar_type() == torch::kInt32, "B must be int32 (int2x16)");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && S.is_contiguous(), "A, B, scale_B must be contiguous");
    TORCH_CHECK(A.device().is_xpu() && B.device() == A.device() && S.device() == A.device(),
            "A, B, scale_B must be on the same XPU");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && S.dim() == 2, "A, B, scale_B must be 2D");
    const int64_t k = A.size(1), n = B.size(1);
    TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128 (scale group size)");
    TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16");
    TORCH_CHECK(B.size(0) == k / 16, "B.size(0) must be K/16");
    TORCH_CHECK(S.size(0) == k / 128 && S.size(1) == n, "scale_B must be [K/128, N]");
}

torch::Tensor out_like(const torch::Tensor &A, int64_t n, const std::optional<torch::Tensor> &C_out) {
    if (!C_out.has_value()) return A.new_empty({A.size(0), n});
    const auto &C = *C_out;
    TORCH_CHECK(C.scalar_type() == A.scalar_type() && C.is_contiguous() && C.device() == A.device(),
            "C must be a contiguous tensor of A's dtype and device");
    TORCH_CHECK(C.dim() == 2 && C.size(0) == A.size(0) && C.size(1) == n, "C must be [M, N]");
    return C;
}

torch::Tensor upcvt(torch::Tensor A, torch::Tensor B, torch::Tensor S, std::optional<torch::Tensor> C_out,
        at::ScalarType dt) {
    check_gemm(A, B, S, dt);
    auto C = out_like(A, B.size(1), C_out);
    ternsycl_upcvt_run(queue_of(A), dt == torch::kBFloat16, A.data_ptr(), B.data_ptr<int32_t>(), S.data_ptr(),
            C.data_ptr(), nullptr, 0, A.size(0), B.size(1), A.size(1));
    return C;
}

}  // namespace

torch::Tensor int2_fp16_upcvt_gemm_run(torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
        std::optional<torch::Tensor> C_out) {
    RECORD_FUNCTION("int2_fp16_upcvt_gemm", {A, B});
    return upcvt(A, B, scale_B, C_out, torch::kHalf);
}

torch::Tensor int2_bf16_upcvt_gemm_run(torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
        std::optional<torch::Tensor> C_out) {
    RECORD_FUNCTION("int2_bf16_upcvt_gemm", {A, B});
    return upcvt(A, B, scale_B, C_out, torch::kBFloat16);
}

// postop 1 = silu(acc) * other (SwiGLU gate), 2 = acc + other (residual).
torch::Tensor int2_fp16_upcvt_gemm_postop_run(torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
        torch::Tensor other, int64_t postop) {
    RECORD_FUNCTION("int2_fp16_upcvt_gemm_postop", {A, B});
    TORCH_CHECK(postop == 1 || postop == 2, "postop must be 1 (silu*other) or 2 (+other)");
    check_gemm(A, B, scale_B, torch::kHalf);
    TORCH_CHECK(other.scalar_type() == torch::kHalf && other.is_contiguous() && other.device() == A.device(),
            "other must be a contiguous fp16 tensor on A's device");
    TORCH_CHECK(other.numel() == A.size(0) * B.size(1), "other must have M*N elements");
    auto C = A.new_empty({A.size(0), B.size(1)});
    ternsycl_upcvt_run(queue_of(A), false, A.data_ptr(), B.data_ptr<int32_t>(), scale_B.data_ptr(), C.data_ptr(),
            other.data_ptr(), (int)postop, A.size(0), B.size(1), A.size(1));
    return C;
}

// int2 x int8 DPAS: A quantized to int8 per (row, 128-group) by a pre-kernel.
torch::Tensor int2_fp16_dpas_gemm_run(torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
        std::optional<torch::Tensor> C_out) {
    RECORD_FUNCTION("int2_fp16_dpas_gemm", {A, B});
    check_gemm(A, B, scale_B, torch::kHalf);
    const int64_t m = A.size(0), k = A.size(1);
    auto C = out_like(A, B.size(1), C_out);
    auto SA = A.new_empty({k / 128 * ternsycl_int8_ldsa((int)m)});
    auto Aq = A.new_empty({m * k}, torch::kChar);
    ternsycl_int8_run(queue_of(A), false, A.data_ptr(), B.data_ptr<int32_t>(), scale_B.data_ptr(), C.data_ptr(),
            SA.data_ptr(), Aq.data_ptr<int8_t>(), m, B.size(1), k);
    return C;
}

// forward: H(s * x) / sqrt(1024) per 1024 block of the last dim; inverse:
// s * H(x) / sqrt(1024). signs: int8 +-1 [K], or none.
torch::Tensor hadamard_fwht_run(torch::Tensor x, std::optional<torch::Tensor> signs, int64_t block, bool inverse) {
    RECORD_FUNCTION("hadamard_fwht", {x});
    TORCH_CHECK(x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16,
            "hadamard_fwht: x must be fp16 or bf16");
    TORCH_CHECK(x.is_contiguous() && x.device().is_xpu(), "hadamard_fwht: x must be a contiguous XPU tensor");
    TORCH_CHECK(block == 1024, "hadamard_fwht: only block 1024 is built");
    const int64_t K = x.size(-1);
    TORCH_CHECK(K % 1024 == 0, "hadamard_fwht: last dim must be a multiple of 1024");
    const int8_t *s = nullptr;
    if (signs.has_value() && signs->defined()) {
        TORCH_CHECK(signs->scalar_type() == torch::kChar, "hadamard_fwht: signs must be int8");
        TORCH_CHECK(signs->is_contiguous() && signs->numel() == K && signs->device() == x.device(),
                "hadamard_fwht: signs must be a contiguous [K] tensor on x's device");
        s = signs->data_ptr<int8_t>();
    }
    auto y = torch::empty_like(x);
    const int64_t rows = x.numel() / K;
    if (rows == 0) return y;
    ternsycl_hadamard_run(queue_of(x), x.scalar_type() == torch::kBFloat16, x.data_ptr(), s, y.data_ptr(), rows,
            (int)K, inverse);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(ternsycl, m) {
    m.def("int2_fp16_upcvt_gemm_run", &int2_fp16_upcvt_gemm_run);
    m.def("int2_fp16_upcvt_gemm_postop_run", &int2_fp16_upcvt_gemm_postop_run);
    m.def("int2_bf16_upcvt_gemm_run", &int2_bf16_upcvt_gemm_run);
    m.def("int2_fp16_dpas_gemm_run", &int2_fp16_dpas_gemm_run);
    m.def("hadamard_fwht_run", &hadamard_fwht_run);
}
