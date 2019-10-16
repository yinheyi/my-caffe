#include <boost/bind.hpp>
#include <glog/logging.h>

#include <signal.h>
#include <csignal>

#include "caffe/util/signal_handler.h"

namespace {
  /* 在信号处理函数中, 对static volatile sig_atomic_t类型的值进行操作，
     然后直接返回, 这是标准的做法。 sig_atomic_t类型的变量保证了即使在
     可能被信号处理程序中断时也保持原子性。 */
  static volatile sig_atomic_t got_sigint = false;
  static volatile sig_atomic_t got_sighup = false;
  static bool already_hooked_up = false;

  /**
    @brief SIGINT和SIGHUP信号的处理程序。 它的动作很简单，也是标准的做法：
          对static volatile sig_atomic_t类型有变量进行操作变变直接返回。
    @param [in] signal 信号的ID.
   */
  void handle_signal(int signal) {
    switch (signal) {
    case SIGHUP:
      got_sighup = true;
      break;
    case SIGINT:
      got_sigint = true;
      break;
    }
  }

  /**
    @brief 该函数为信号SIGINT和信号SIGHUP绑定相应的信号处理函数。看这部分时，
    要了解一下信号相应的知识，可以看一下unix核心编程一书。
    */
  void HookupHandler() {
    if (already_hooked_up) {
      LOG(FATAL) << "Tried to hookup signal handlers more than once.";
    }
    already_hooked_up = true;

    struct sigaction sa;
    // Setup the handler
    sa.sa_handler = &handle_signal;
    // Restart the system call, if at all possible
    sa.sa_flags = SA_RESTART;
    // Block every signal during the handler
    sigfillset(&sa.sa_mask);
    // Intercept SIGHUP and SIGINT
    if (sigaction(SIGHUP, &sa, NULL) == -1) {
      LOG(FATAL) << "Cannot install SIGHUP handler.";
    }
    if (sigaction(SIGINT, &sa, NULL) == -1) {
      LOG(FATAL) << "Cannot install SIGINT handler.";
    }
  }

  // Set the signal handlers to the default.
  // 与上面函数相反，恢复默认的信号处理函数。
  void UnhookHandler() {
    if (already_hooked_up) {
      struct sigaction sa;
      // Setup the sighup handler
      sa.sa_handler = SIG_DFL;
      // Restart the system call, if at all possible
      sa.sa_flags = SA_RESTART;
      // Block every signal during the handler
      sigfillset(&sa.sa_mask);
      // Intercept SIGHUP and SIGINT
      if (sigaction(SIGHUP, &sa, NULL) == -1) {
        LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
      }
      if (sigaction(SIGINT, &sa, NULL) == -1) {
        LOG(FATAL) << "Cannot uninstall SIGINT handler.";
      }

      already_hooked_up = false;
    }
  }

  /* 每一次执行该函数时，不仅仅返回got_sigint的值，还会把它再置为false,意思就是
     每当调用该函数时，实际上是在检测自从上次调用该函数到现在，got_sigint是否被
     置为了true.
   */
  bool GotSIGINT() {
    bool result = got_sigint;
    got_sigint = false;
    return result;
  }

  // Return true iff a SIGHUP has been received since the last time this
  // function was called.
  bool GotSIGHUP() {
    bool result = got_sighup;
    got_sighup = false;
    return result;
  }
}  // namespace

namespace caffe {

SignalHandler::SignalHandler(SolverAction::Enum SIGINT_action,
                             SolverAction::Enum SIGHUP_action):
  SIGINT_action_(SIGINT_action),
  SIGHUP_action_(SIGHUP_action) {
  HookupHandler();    // 绑定相应的信号处理函数。
}

SignalHandler::~SignalHandler() {
  UnhookHandler();  // 恢复至默认的信号处理函数。
}

/** @brief 根据相应的值，返回要执行的动作。 */
SolverAction::Enum SignalHandler::CheckForSignals() const {
  if (GotSIGHUP()) {
    return SIGHUP_action_;
  }
  if (GotSIGINT()) {
    return SIGINT_action_;
  }
  return SolverAction::NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallback SignalHandler::GetActionFunction() {
  // 这是使用了boost::bind函数，把一个类成员函数(它有一个this指针）包装成0个参数的函数对象。
  return boost::bind(&SignalHandler::CheckForSignals, this);
}

}  // namespace caffe
