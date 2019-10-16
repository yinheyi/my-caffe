#ifndef INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
#define INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_

#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

namespace caffe {

/**
  @brief 定义了一个操作系统的信号处理类，它截获SIGINT和SIGHUP信号，并执行
  相应的信号处理函数。
  */
class SignalHandler {
 public:
  /**
    @brief 该构造函数指定了收到SIGINT和SIGHUP信号要执行什么动作(stop/snapshot/none)，
    并且绑定了SIGINT和SIGHUP相应的信号处理函数。
    */
  SignalHandler(SolverAction::Enum SIGINT_action,
                SolverAction::Enum SIGHUP_action);

  /** @brief 在析构函数中，会把SIGINT和SIGHUP的信号处理函数恢复为默认的信号处理函数.  */
  ~SignalHandler();

  /** @brief  该函数返回一个调用对象(ActionCallback是在solver.cpp文件中定义的。). 该函数
    对象是一个返回SolverAction::Enum类型的参数为空的函数。意思就是说调用该函数返回的函数对象
    就可以知道要执行什么样的Action.
    */
  ActionCallback GetActionFunction();
 private:
  SolverAction::Enum CheckForSignals() const;
  SolverAction::Enum SIGINT_action_;
  SolverAction::Enum SIGHUP_action_;
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
