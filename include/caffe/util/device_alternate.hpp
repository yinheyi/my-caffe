/**
  @file device_alternate.hpp
  @details 该文件定义了在CPU_ONLY模式下或者普通模式下一些常使用到的代码的宏定义。
  */

#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CPU_ONLY  // CPU-only Caffe.

#include <vector>

// 在CPU_ONLY模式下，只要出现了GPU相关功能调用时，就给出致命的错误信息提示。 
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

// 阻止Layer中与GPU相关的前向传播函数和反向传播函数。
#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

// 阻止Layer中与GPU相关的前向传播函数。
#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

// 阻止Layer中与GPU相关的反向传播函数。
#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef USE_CUDNN  // cuDNN acceleration library.
#include "caffe/util/cudnn.hpp"
#endif

//
// CUDA macros
//

/** 
  @brief 对与CUDA相关的各种功能函数的调用的返回值的检测，执行成功时，返回值为CudaSucess。
  如果执行失败时，会输出错误信息。这里使用了google glog库的CHECK_EQ宏定义。  
  下面的宏定义中，使用do {} while(0) 来实现在一个宏中定义多条语句代码, 这是一个技巧。
  */
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

// CUDA: library error reporting.  (给定错误码，返回相应的字符串描述信息。)
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

/** 
  @brief给定线程数，返回需要的block块的数目。
  @param [in] n 线程数
  */
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe

#endif  // CPU_ONLY

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
