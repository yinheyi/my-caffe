#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

  /**
    @brief 该枚举类型用于一个函数的返回值，表示用于当收到SIGINT信号或SIGHUP信号时，
    Caffe会做一些什么事情.具体要去看一个使用到的地方就明白了。
    */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  virtual ~Solver() {}
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  /**
    @brief 设置获取Action的回调函数的指针给成员变量。
    @param [in] func 要设置的回调函数。
    */
  void SetActionFunction(ActionCallback func);

  /** @brief 调用的设置的回调函数，获取需要执行的Action.  */
  SolverAction::Enum GetRequestedAction();


  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string& resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);


  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  // 它会调用RestoreSolverStateFromHDF5()函数或者RestoreSolverStateFromBinaryProto()函数。
  // 至于调用哪个函数，由文件名的后缀决定。
  void Restore(const char* resume_file);

  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.

  // 第一步，把学习到的net进行snapshot.(本类已经实现)
  // 第二步：把SolverState进行snapshot.(子类需要实现)
  void Snapshot();
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() const { return iter_; }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  /** @brief 当需要进行snapshot时，核查指定目录是否有写权限。 */
  void CheckSnapshotWritePermissions();

  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;

 protected:
  /**
    @brief 获取snapshot保存的文件名.
    @param [in] externsion为保存文件名的扩展名。
    @detail 文件名为：snapshot_prefix()+"_iter_"+迭代次数.
    */
  string SnapshotFilename(const string& extension);

  /**
    @brief 把net当前的状态snapshot到NetParameter对应的二进制文件中。
    @return 返回snapshot之后的文件名,它是以.caffemodel为后缀的文件。
   */
  string SnapshotToBinaryProto();

  /**
    @brief 把net当前的状态snapshot到HDF5文件中. 
    @return 返回snapshot之后的文件名,它是以.h5为后缀的文件。
   */
  string SnapshotToHDF5();

  /** @brief 该函数分别调用Test()函数测试所有的test网络。*/
  void TestAll();

  /**
    @brief 测试给定的test网络。
    @param [in] test_net_id 指定测试第几个test_net, 默认为第0个。 */
  void Test(const int test_net_id = 0);

  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);

  /**
    @brief 功能描述：使用本次迭代求出来的loss更新smoothed_loss的值。
    @param [in] loss         本次迭代求出来的loss的值。
    @param [in] start_iter   开始时的迭代次数值. 它与当前iter_的值相结合，用于计算出当前的loss值要
                             替换的最旧的loss的值的下标。
    @param [in] average_loss 求smoothed_loss_时，表示使用最新的几次的loss值，也就是一个窗口的大小。
    */
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;
  int iter_;                                    //< 当前进行到的迭代次数。
  int current_step_;                            //< 它表示
  shared_ptr<Net<Dtype> > net_;                 //< solver对应的训练的网络，即train_net, 只有一个。
  vector<shared_ptr<Net<Dtype> > > test_nets_;  //< solver对应的测试的网络， 它可以有多个，所以是一个vector.
  vector<Callback*> callbacks_;
  vector<Dtype> losses_;                        //< 它里面存放了最新的n次迭代时的loss的值。
  Dtype smoothed_loss_;                         //< n次迭代的平均值的loss值。
  bool requested_early_exit_;                   //< 在训练过程中是否需要进行早停。

  ActionCallback action_request_function_;      //< 它是一个函数指针，用于获取当前用户想要执行的action,比如 stop,snapshot.
                                                //< 看一个util/signal_handle.hpp文件会更明白它的用途的。

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteretion_timer_;
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
