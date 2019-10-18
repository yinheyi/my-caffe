#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "caffe/util/device_alternate.hpp"

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// 禁用类的复制和赋值构造函数的类
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// 使用double和float类型实例化一个类模板
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

// 使用double和float类型实例化gpu的前向传播函数
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);

// 使用double和float类型实例化gpu的反向传播函数
#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu( \
      const std::vector<Blob<float>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<float>*>& bottom); \
  template void classname<double>::Backward_gpu( \
      const std::vector<Blob<double>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// 定义没有实现的宏，实现的功能是：会输出一条fatal的日志。
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv { class Mat; }

namespace caffe {

// 使用boost库中的shared_ptr代替c++11中的shared_ptr, 原因是CUDA兼容性的问题.
using boost::shared_ptr;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

// 全局的初始化函数，目前该函数会初始化google flags 和 google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
/** @brief 下面定义了一个全局的单例类caffe.   */
class Caffe {
 public:
  ~Caffe();

  /** @brief 获取单例实体对象, 它返回一个引用。 */
  static Caffe& Get();

  // 定义了一个枚举类型
  enum Brew { CPU, GPU };

  /**
    @brief 定义了一个RNG的类，该类的唯一一个作用就是返回一个随机数生成器: 要么是一个函数指针,
    要么是一个重载了()操作符的类指针.

    实际上，该类的具体工作交给了类内部定义的另一个Generator来去完成。而Generator类的具体工作
    呢，又是交给了boost库中的boost::mt19937类来完成。
   */
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    /** 返回一个随机数生成器, 看看它的返回值是一个void类型的指针，之所以是void类型是因为不能
      底层实现中的随机数生成器是什么类型的，但是可以肯定是函数指针或重载了()的类指针。*/
    void* generator();

   private:
    // 实现工作是交给了Generator类来完成的。
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifndef CPU_ONLY
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);
  // Parallel training
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static int solver_rank() { return Get().solver_rank_; }
  inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
  inline static bool multiprocess() { return Get().multiprocess_; }
  inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
  inline static bool root_solver() { return Get().solver_rank_ == 0; }

 protected:
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif
  shared_ptr<RNG> random_generator_;
  Brew mode_;

  // Parallel training
  int solver_count_;
  int solver_rank_;
  bool multiprocess_;

 private:
  // 在单例类中，构造函数都是私有的。
  Caffe();
  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
