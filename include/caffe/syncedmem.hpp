/**
  @file syncedmem.hpp
  @brief caffe的内存管理文件。

  1. 在该文件内定义了两个全局的内存申请和释放的内联函数。
  2. 定义了一个cpu与gpu数据的同步类，用于管理相应的数据内存。
  */
#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

/* 涉及到一个MKL库， intel mathmeknel library, 包含了矩阵相关/线性代数相关的数学运算等。
   该库绝对值得了解学习一下。 官网 https://software.intel.com/en-us/mkl/ */
#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

/**
  @brief 功能描述： caffe内的内存申请函数,负责申请指定大小的内存。
  @param [out] ptr       它是一个指针的指针，用于存放申请到的内存指针。
  @param [in]  size      要申请内存的字节数。
  @param [out] use_cuda  它是一个bool类型的指针，用于保存本次的内存申请是否是使用的cuda的
                         内存申请函数来申请的。
  @return 返回值为空。

  具体来说，分为三种情况：  
   1. 当使用gpu模式时，使用cuda的内存申请函数来申请内存。
   2. 当使用MKL库时，使用MLK库内的内存申请函数来申请内存。
   3. 否则的话，就使用malloc来申请内存。
  */
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

/**
  @brief 功能描述：与CaffeMallocHost功能相反，它进行内存的释放。
  @param [in] ptr 内存的指针
  @param [in] use_cuda 指明要释放的内存是否是由cuda的函数申请的。
  @return 返回值为空。
  */
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
  @brief caffe中内存的管理类，主要负责了内存和显示的申请释放, 数据在cpu内存和gpu显之间
  的同步。

  对于该类：要知道：
  1. cpu上的数据内存是由CaffeMallocHost函数和CaffeFreeHost函数来申请和释放。
  2. gpu显存上的数据是由cudaMalloc函数和cudaFree函数来申请和释放。
  3. 通过to_cpu函数和to_gpu函数来保持数据在内存和显存之间的同步。
 */
  /
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();

  /* 该枚举类型定义了某一时刻数据在cpu与gpu的状态：
     1. 未初始化状态   2.cpu上数据是最新的  3.gpu上最新的， 4. 同步状态，*/
  enum SyncedHead
  {
      UNINITIALIZED,
      HEAD_AT_CPU,
      HEAD_AT_GPU,
      SYNCED
  };

  /** @brief 功能描述：获取cpu内存区数据的指针, 不应该通过该函数返回的指针修改数据,
       否则的话，会造成cpu与gpu数据的不同步 */
  const void* cpu_data();
  /** @brief 功能描述：获取cpu内存区数据的指针, 可以通过该函数返回的指针修改数据。
      原因是因为：该函数会把数据置为cpu有效。*/
  void* mutable_cpu_data();
  /** @brief 功能描述：设置cpu内存区数据.
      @param [in] data 要设置数据的指针 */
  void set_cpu_data(void* data);
  const void* gpu_data();
  void* mutable_gpu_data();
  void set_gpu_data(void* data);
  /** @brief 功能描述： 该函数返回数据在cpu和gpu上的状态。  */
  SyncedHead head() const { return head_; }
  /** @brief 功能描述： 该函数返回数据的字节大小 */
  size_t size() const { return size_; }
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();
  /** @brif 功能描述：该函数负责把同步数据至CPU内存区上。根据数据不同的状态做不同处理：  
      1. 如果数据是未初始化状态，则会申请内存, 并初始为0.
      2. 如果数据是在CPU 区的状态，则不需要处理.
      3. 如果数据是在GPU区的状态，则复制到CPU内存区.
      4. 如果数据是CPU和GPU同步状态，则不需要处理.  */
  void to_cpu();
  /** @brif 功能描述：该函数负责把同步数据至gpu内存区上。根据数据不同的状态做不同处理：  
      1. 如果数据是未初始化状态，则会申请内存, 并初始为0.
      2. 如果数据是在CPU区的状态，则复制到GPU内存区.
      3. 如果数据是在GPU 区的状态，则不需要处理.
      4. 如果数据是CPU和GPU同步状态，则不需要处理.  */
  void to_gpu();

  void* cpu_ptr_;             // cpu数据内存的指针
  void* gpu_ptr_;             // gpu显存的指针
  bool own_cpu_data_;         /* 指明该类对象是否拥有[成员变量cpu_ptr]指向的cpu内存数据,
                                 如果拥有的话，就要对该内存空间何时释放进行管理。*/
  bool own_gpu_data_;
  SyncedHead head_;           // 数据在cpu和gpu上的状态
  size_t size_;               // 数据的字节数
  bool cpu_malloc_use_cuda_;  // 指明该类管理的cpu内存是否是由cuda来申请的。
  int device_;                 // gpu的设备号。

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
