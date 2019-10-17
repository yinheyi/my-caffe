#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {
/**
  @brief 功能描述：该函数模板计算两个矩阵的相乘(general matrix matrix).   
      矩阵C = Alpha * op(矩阵A) × op(矩阵B) + beta * 矩阵C.
  @param [in] TransA 表示矩阵A是否进行转置, 当转置时，op(矩阵A） = 矩阵A', 否则op(矩阵A) = 矩阵A 
  @param [in] TransB 表示矩阵B是否进行转置。
  @param [in] M      表示op(矩阵A)的行,或者矩阵C的行。
  @param [in] N      表示op(矩阵B)的列,或者矩阵C的列。
  @param [in] K      表示op(矩阵A)的列,或者op(矩阵B)的行。
  @param [in] alpha  标量alpha
  @param [in] A      矩阵A的地址
  @param [in] B      矩阵B的地址
  @param [in] beta   标量beta
  @param [in] C      矩阵B的地址
  */
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

/**
  @brief 功能描述：该函数模板实现了矩阵与矢量的相乘运算(general matrix vector)  
     矢量y = alpha * op(矩阵A) × 矢量x  + beta * 矢量y
  @param [in] TransA 表示矩阵A是否进行转置。
  @param [in] M      表示矩阵A的行(与gemm函数中的含义不同)
  @param [in] N      表示矩阵A的列
  @param [in] alpha  标量alpha
  @param [in] A      矩阵A的地址
  @param [in] x      矢量x的地址
  @param [in] beta   标量beta
  @param [in] y      矢量y的地址
  */
template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

/**
  @brief 功能描述：该函数模板实现如下的矢量运算：  
      矢量Y = alpha * 矢量X + 矢量Y
  @param [in] N      矢量X或矢量Y的元素个数。
  @param [in] alpha  标量alpha
  @param [in] X      矢量X的地址
  @param [in,out] Y  矢量Y的地址
  */
template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

/**
  @brief 功能描述：与函数caffe_axpy功能类似，实现如下的矢量运算：  
      矢量Y = alpha * 矢量X + beta * 矢量Y
  @param [in] N      矢量X或矢量Y的元素个数。
  @param [in] alpha  标量alpha
  @param [in] X      矢量X的地址
  @param [in] beta   标量alpha
  @param [in,out] Y  矢量Y的地址
  */
template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

/**
  @brief 功能描述：实现数据的拷贝，从X拷贝到Y中。
  @param [in] N 要拷贝的数据项数。
  @param [in] X 源数据的地址。
  @param [out] Y 目的数据的地址。
  */
template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

/**
  @brief 功能描述： 在指定的地址处设置指定的值。
  @param [in] N     要设置的数据项数。
  @param [in] alpha 要设置的值。
  @param [out] Y    指定数据的地址。
  */
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

/**
  @brief 功能描述：在矢量中每一个元素上添加一个标量：
   矢量X = 矢量X + alpha + [1,1, 1, .....1]' 
  @param [in] N     矢量的元素个数。
  @param [in] alpha 要增加的标题的值。
  @param [in,out] X 矢量X的地址。
  */
template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

/**
  @brief 功能描述：对一个矢量中每一个元素乘上一个标量：
   矢量X = alpha * 矢量X
  @param [in] N     矢量的元素个数。
  @param [in] alpha 标量alpha
  @param [in,out] X 矢量X的地址。
  */
template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

/**
  @brief 功能描述：实现矢量的相加运算:  
  矢量Y = 矢量a + 矢量b
  */
template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

/**
  @brief 功能描述：实现矢量的相减运算:  
  矢量Y = 矢量a - 矢量b
  @param [in]  N  矢量中元素的个数。
  @param [in]  a  矢量a的地址
  @param [in]  b  矢量b的地址
  @param [out] y  矢量y的地址
  */
template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

/**
  @brief 功能描述：实现矢量与矢量的elementwise的相乘。  
  矢量Y = 矢量a * 矢量b
  @param [in]  N  矢量中元素的个数。
  @param [in]  a  矢量a的地址
  @param [in]  b  矢量b的地址
  @param [out] y  矢量y的地址
  */
template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

/**
  @brief 功能描述：实现矢量与矢量的elementwise的相除。  
  矢量Y = 矢量a ./ 矢量b
  @param [in]  N  矢量中元素的个数。
  @param [in]  a  矢量a的地址
  @param [in]  b  矢量b的地址
  @param [out] y  矢量y的地址
  */
template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

/**
  @brief 功能描述：实现矢量的elementwise的求指数幂
  矢量Y = (矢量a) ^ b
  @param [in]  N  矢量中元素的个数。
  @param [in]  a  矢量a的地址
  @param [in]  b  标量b的地址
  @param [out] y  矢量y的地址
  */
template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

/**
  @brief 功能描述：实现矢量的elementwise的求平方
  矢量Y = 矢量a * 矢量a
  @param [in]  N  矢量中元素的个数。
  @param [in]  a  矢量a的地址
  @param [out] y  矢量y的地址
  */
template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

/**
  @brief 功能描述：实现矢量的elementwise的求平方根: 矢量Y = sqrt(矢量a)
  @param [in]  N  矢量中元素的个数。
  @param [in]  a  矢量a的地址
  @param [out] y  矢量y的地址
  */
template <typename Dtype>
void caffe_sqrt(const int N, const Dtype* a, Dtype* y);

/** @brief 功能描述： 返回随机整数 */
unsigned int caffe_rng_rand();

/** @brief 功能描述： 返回下一个可以表示的大于b的数。
    @param [in] 当前的数值。

    什么意思呢？比如当前数是12.1, 调用该函数就返回在计算机中能表示的下一个大于12.1数.
    因为在计算机中数字都是二进制表示的。 了解一个一下float类型和double类型的数字在内存
    中的表示方法就明白了。
 */
template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

/**
  @brief 功能描述： 生成n个在区间[a,b]之间的服从均匀分布的数。
  @param [in] n    要生成的数的个数。
  @param [in] a,b  用于确定均匀分布的区间的上下边界。
  @param [out] r   用于保存结果的指针。
  */
template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

/**
  @brief 功能描述： 生成n个在服从均值为mu, 方差为sigma的正态分布分的数。
  @param [in] n      要生成的数的个数。
  @param [in] mu     正态分布密度函数的均值
  @param [in] sigma  正态分布密度函数的方差
  @param [out] r     用于保存结果的指针。
  */
template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype* r);

/**
  @brief 功能描述： 生成n个在服从概率为p的bernoulli分布的数。（即0-1分布，为1的概率为p）
  @param [in] n      要生成的数的个数。
  @param [in] p
  @param [out] r     用于保存结果的指针。
  */
template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

/**
  @brief 功能描述： 对矢量a的每一个元素进行elementwise的e指数运算。
      矢量Y = e^(矢量Y)
  */
template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

/**
  @brief 功能描述： 对矢量a的每一个元素进行elementwise的e对数运算。
      矢量Y = log(矢量Y)
  */
template <typename Dtype>
void caffe_log(const int n, const Dtype* a, Dtype* y);

/**
  @brief 功能描述： 求矢量a的每一个元素的绝对值。
      矢量Y = |矢量Y|
  */
template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);

/**
  @brief 功能描述： 矢量x与矢量Y的点积操作。
     标量result = 矢量x .* 矢量y.
  */
template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

/**
  @brief 功能描述： 矢量x与矢量Y的点积操作, 增加了步长参数，意思就是每隔给定的步长进行一次元素相乘。
  @param [in] n   矢量点积相乘的总元素个数。
  @param [in] x   矢量X的地址
  @param [in] x   矢量X的步长，(矢量X中的元素个数一定要大于或等于1 + (n-1) * incx )
  @param [in] y   矢量Y的地址
  @param [in] y   矢量Y的步长，(矢量Y中的元素个数一定要大于或等于1 + (n-1) * incy )
  */
template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

/**
  @brief 功能描述： 求矢量X中每一个元素的绝对值之和。
  */
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

/**
  @brief 求给定一个数的正负号，如果为正返回1,如果为0返回0, 如果为负返回-1.
  @return 返回值为-1,0或1.
  */
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
/**
  @brief 定义了一个很有意思的宏, 该宏定义了一个函数，具体看代码。
  */
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// 该宏定义了一个名为caffe_cpu_sign的函数，用于求给定矢量中每一个元素的符号。
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.

// 该宏定义了一个名为caffe_cpu_sgnbit的函数，用于求给定矢量中每一个元素是否为负号，
// 如果是则置true,不是负数，则置为false.(具体看看signbit函数的返回值就知道了。)
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])))

// 该宏定义了一个名为caffe_cpu_fabs的函数，用于求给定矢量中每一个元素的绝对值。
DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))

/**
  @brief 功能描述: 实现 矢量Y = alpha * 矢量X的功能。
  */
template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
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
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype* X, cudaStream_t str);
#endif

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sqrt(const int n, const Dtype* a, Dtype* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
