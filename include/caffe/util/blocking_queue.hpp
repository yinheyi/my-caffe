#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>

namespace caffe {

/**
  @brief 定义了一个阻塞队列的管理类, 底层是调用std中的queue来实现，在此
  基础上增加了同步机制, 保证多线程的安全有效性。
  */
template<typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue();

  /** 
    @brief 功能描述：把对象加入的队列中, 并通知可能等待其它线程现在队列内又有新来的内容了。
    @param [in] t 要加入到队列中的元素。
   */
  void push(const T& t);

  /**
    @brief 尝试从队列中弹出元素.
    @param [in] t 用于存放弹出元素的指针。
    @return 如果队列为空，则返回false, 如果队列不为空，则返回true.
    */
  bool try_pop(T* t);

  /**
    @brief 出队的函数接口，如果队列为空，则线程就会被阻塞等待。
    @param [in] log_on_wait 这是一个string字符串，用于在阻塞等待时每隔1000次打印一个日志提示信息。
    这种情况会出现在pop的线程远远大于push的线程的情况，因为每当push一个元素时，就可能被其它线程拿
    走了。
    @return 返回值是被pop出的元素。
    */
  T pop(const string& log_on_wait = "");

  /**
    @brief 该函数尝试获到队列头的元素的值 ，但是不会出队。
    @param [in] t 该指针用于存放获取到的元素。
    @return 当队列不为空是返回true, 否则返回false.
    */
  bool try_peek(T* t);

  /** @brief 获取队列头的元素，但是不会出队，其它与pop相同。该函数也可能会被阻塞。 */
  T peek();

  /** @brief 返回队列中的元素个数。 */
  size_t size() const;

 protected:
  std::queue<T> queue_;       // 底层使用std::queue来实现队列
  class sync;
  shared_ptr<sync> sync_;    // 同步类对象，它里面有一个mutex和condition_variable来保证多线程的同步和通信。

DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace caffe

#endif
