#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

// 包含该头文件，为了使用boost库中的data_time库中的ptime类。
#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

/**
  @brief 定义了一个计时器的类，使用boost库内的API boost::posix_time::microsec_clock::local_time()
  来获取当前local时间的clock的微秒值信息保存在ptime类对象(也是boost库中定义的类)中。
  */
class Timer {
 public:
  Timer();
  virtual ~Timer();

  /** 
    @brief 计时开始函数. 如果多次执行该函数，后面多余的执行不做任何事,意思就是说它只会存放
    第一次开始的时间值。 
   */
  virtual void Start();

  /** 
    @brief 计时停止函数. 如果多次执行该函数，后面多余的执行不做任何事, 意思就是说它只会存放
    第一次停止的时间值。
   */
  virtual void Stop();

  /** @brief 该函数获取经过的秒数, 如果计时器没有停止，先停止再求值。*/
  virtual float Seconds();
  /** @brief 该函数获取经过的毫秒数, 如果计时器没有停止，先停止再求值。*/
  virtual float MilliSeconds();
  /** @brief 该函数获取经过的微秒数, 如果计时器没有停止，先停止再求值。*/
  virtual float MicroSeconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;        // 该类是否执行了Init()函数进行初始化过了。
  bool running_;        // 当前类是否正处于计时状态，即start_cpu_变量内已经存取了起始时间的信息了。
  bool has_run_at_least_once_;    // 只要执行过一次Start()函数，就会把它置为true, 说明start_cpu_成员变量已经存放了有意义的时间信息。

#ifndef CPU_ONLY
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
  boost::posix_time::ptime start_cpu_;   // 使用boost库中ptime类定义了一个time对象, 用于存放开始时间信息
  boost::posix_time::ptime stop_cpu_;    // 使用boost库中ptime类定义了一个time对象, 用于存放停止时间信息

  float elapsed_milliseconds_;   // 起始时间到停止时间之间经过的毫秒。
  float elapsed_microseconds_;   // 起始时间到停止时间之间经过的微秒。
};

/**
  @brief 该类继承自timer类，但是仅仅使用一下它的成员变量，重载了下面的几个函数. 
  与Timer类相比，把与GPU相关的代码给删除掉了。
  */
class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
