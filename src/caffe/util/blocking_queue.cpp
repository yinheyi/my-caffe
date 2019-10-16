#include <boost/thread.hpp>
#include <string>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

 /**
   @brief 定义了一个用于同步机制的类， 它只包含了两个公有的成员变量,没有任何成员函数。

   具体来说，这里涉及到一个boost中与线程或和锁相关的知识：
   boost::mutex类是一个互斥体，用于多线程中保证数据的同步.
   boost::condition_variable 是条件变量，与boost::mutex一起使用，用于多个线程之间的通信，
   即：一个线程因为等待某一个资源被阻塞了，当其它线程产生了这个资源之后就可以通过condition_variable
   来焕醒被阻塞的线程了。
  */
template<typename T>
class BlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {

  // scoped_lock类是对mutex的封装，提供了RAII的机制，即在构造函数中获取锁，在
  // 析构函数中释放锁。
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();

  // 焕醒一个被阻塞的其它线程，告诉它队列非空了，可以执行干它的事情了。
  sync_->condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {

    // 打印一个日志信息，这里表明了当前队列的push的线程远远小于pop的线程。
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    // 获取不到资源，则阻塞。
    sync_->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;

}  // namespace caffe
