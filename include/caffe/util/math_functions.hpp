#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <cmath> // for std::fabs and std::signbit
#include <stdint.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

/**
 * @brief \b C=alpha*A*B+beta*C
 *
 * @tparam Dtype
 * @param TransA 是否要对A做转置操作
 * @param TransB 是否要对B做转置操作
 * @param M A、C 的行数
 * @param N B、C 的列数
 * @param K A 的列数， B 的行数
 * @param alpha
 * @param A 输入矩阵（一维数组格式）
 * @param B 输入矩阵（一维数组格式）
 * @param beta
 * @param C 输入矩阵（一维数组格式）
 */
// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, Dtype *C);

/**
 * @brief  \b y=alpha*A*x+beta*y
 * @tparam Dtype
 * @param TransA 是否要对A做转置操作
 * @param M A 的行数
 * @param N A 的列数
 * @param alpha
 * @param A
 * @param x
 * @param beta
 * @param y
 */
template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype *A, const Dtype *x,
                    const Dtype beta, Dtype *y);

/**
 * @brief \b Y=alpha*X+Y
 * @tparam Dtype
 * @param N 为X和Y中element的个数
 * @param alpha
 * @param X
 * @param Y
 */
template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

/**
 * @brief Scales two vectors, adds them to one another and stores result in the
 * vector. <br> \b y := a*x + b*y
 * @tparam Dtype
 * @param N size of the vector
 * @param alpha a
 * @param X x
 * @param beta b
 * @param Y y
 */
template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

/**
 * @brief Copy X to Y
 * @tparam Dtype
 * @param N number of blocks
 * @param X
 * @param Y
 */
template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

/**
 * @brief 用常数 alpha 对 X 进行初始化
 * @tparam Dtype
 * @param N
 * @param alpha
 * @param X
 */
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void *X) {
  memset(X, alpha, N); // NOLINT(caffe/alt_fn)
}

/**
 * @brief 给 X 的每个 element 加上常数 alpha
 * @tparam Dtype
 * @param N
 * @param alpha
 * @param X
 */
template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

/**
 * @brief X = alpha*X
 * @tparam Dtype
 * @param N
 * @param alpha
 * @param X
 */
template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

/**
 * @brief element-wise sqrt
 * @tparam Dtype
 * @param N
 * @param a
 * @param y
 */
template <typename Dtype> void caffe_sqr(const int N, const Dtype *a, Dtype *y);

/**
 * @brief element-wise sqrt
 * @tparam Dtype
 * @param N
 * @param a
 * @param y
 */
template <typename Dtype>
void caffe_sqrt(const int N, const Dtype *a, Dtype *y);

/**
 * @brief element-wise add
 * @tparam Dtype
 * @param N
 * @param a
 * @param b
 * @param y
 */
template <typename Dtype>
void caffe_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

/**
 * @brief element-wise sub
 * @tparam Dtype
 * @param N
 * @param a
 * @param b
 * @param y
 */
template <typename Dtype>
void caffe_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

/**
 * @brief element-wise mul
 * @tparam Dtype
 * @param N
 * @param a
 * @param b
 * @param y
 */
template <typename Dtype>
void caffe_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

/**
 * @brief element-wise div
 * @tparam Dtype
 * @param N
 * @param a
 * @param b
 * @param y
 */
template <typename Dtype>
void caffe_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

/**
 * @brief element-wise pow <br>
 * y[i] = a[i] ^ b
 * @tparam Dtype
 * @param n
 * @param a
 * @param b
 * @param y
 */
template <typename Dtype>
void caffe_powx(const int n, const Dtype *a, const Dtype b, Dtype *y);

/**
 * @brief 返回一个随机数
 * @return 随机数
 */
unsigned int caffe_rng_rand();

/**
 * @brief 返回 b 最大方向上可以表示的最接近的数值。
 * @tparam Dtype
 * @param b
 * @return
 */
template <typename Dtype> Dtype caffe_nextafter(const Dtype b);

/**
 * @brief 产生指定范围内的均匀分布随机数；
 * @tparam Dtype
 * @param n size
 * @param a low
 * @param b high
 * @param r
 */
template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

/**
 * @brief 产生高斯分布随机数
 * @tparam Dtype
 * @param n
 * @param mu
 * @param sigma
 * @param r
 */
template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype *r);

/**
 * @brief 产生伯努利分布随机数
 * @tparam Dtype
 * @param n
 * @param p
 * @param r
 */
template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int *r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int *r);

template <typename Dtype> void caffe_exp(const int n, const Dtype *a, Dtype *y);

template <typename Dtype> void caffe_log(const int n, const Dtype *a, Dtype *y);

template <typename Dtype> void caffe_abs(const int n, const Dtype *a, Dtype *y);

/**
 * @brief 计算步长为1的内积
 * @tparam Dtype
 * @param n
 * @param x
 * @param y
 * @return
 */
template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype *x, const Dtype *y);

/**
 * @brief 计算指定步长的内积
 * @tparam Dtype
 * @param n
 * @param x
 * @param incx
 * @param y
 * @param incy
 * @return
 */
template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype *x, const int incx,
                            const Dtype *y, const int incy);

/**
 * @brief the sum of the absolute values of the elements of vector x
 * @tparam Dtype
 * @param n
 * @param x
 * @return
 */
template <typename Dtype> Dtype caffe_cpu_asum(const int n, const Dtype *x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename Dtype> inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation)                           \
  template <typename Dtype>                                                    \
  void caffe_cpu_##name(const int n, const Dtype *x, Dtype *y) {               \
    CHECK_GT(n, 0);                                                            \
    CHECK(x);                                                                  \
    CHECK(y);                                                                  \
    for (int i = 0; i < n; ++i) {                                              \
      operation;                                                               \
    }                                                                          \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit,
                            y[i] = static_cast<bool>((std::signbit)(x[i])))

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

#ifndef CPU_ONLY // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, Dtype *C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype *A, const Dtype *x,
                    const Dtype beta, Dtype *y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void *X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N)); // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

#ifndef CPU_ONLY
template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X, cudaStream_t str);
#endif

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype *a, const Dtype b, Dtype *y);

template <typename Dtype>
void caffe_gpu_sqrt(const int n, const Dtype *a, Dtype *y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int *r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype *r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int *r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype *x, const Dtype *y, Dtype *out);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_sign(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation)                 \
  template <typename Dtype>                                                    \
  __global__ void name##_kernel(const int n, const Dtype *x, Dtype *y) {       \
    CUDA_KERNEL_LOOP(index, n) { operation; }                                  \
  }                                                                            \
  template <>                                                                  \
  void caffe_gpu_##name<float>(const int n, const float *x, float *y) {        \
    /* NOLINT_NEXT_LINE(whitespace/operators) */                               \
    name##_kernel<float>                                                       \
        <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y);            \
  }                                                                            \
  template <>                                                                  \
  void caffe_gpu_##name<double>(const int n, const double *x, double *y) {     \
    /* NOLINT_NEXT_LINE(whitespace/operators) */                               \
    name##_kernel<double>                                                      \
        <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y);            \
  }

#endif // !CPU_ONLY

} // namespace caffe

#endif // CAFFE_UTIL_MATH_FUNCTIONS_H_
