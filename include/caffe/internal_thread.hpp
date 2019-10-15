#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

 /**
   @brief 该类就是对boost库内的thread的封装，这样一来，该类的子类就可以
   生成一个线程来执行定义好的函数(InternalThreadEntry). 这样很方便的。
   */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
    @brief 线程的开始函数.  在caffe中生成的每一个线程都有一个独立的Caffe对象类的实
    例(见common.h/common.cpp文件).  新创建的Caffe对象的成员变量是使用当前线程中值来
    初始化的.
   */
  void StartInternalThread();

  /** @brief 中断子线程，并等待子线程结束，然后才返回。 */
  void StopInternalThread();

  bool is_started() const;

 protected:
  /** @brief 子类中实现该函数，新创建的线程执行该函数的。 */
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed,
      int solver_count, int solver_rank, bool multiprocess);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
