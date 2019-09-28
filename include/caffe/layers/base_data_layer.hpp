#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/** @brief BaseDataLayer类定义了所有data类的基类，data类负责把数据块blobs
    Net, 应该是net中的InputLayer吧？？？
  */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);

  /** @brief 功能描述： 当前layer层的setup函数， 它会做一些通用的数据层的
      设置的动作(设置成员变量output_labels_和transform_param_的值), 然后调
      用每一个子类定义的DataLayerSetup函数来设置一些
      specifical的setup 功能。
    */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 功能描述： 该函数应该由具体的子类来根据具体的data来实现. */
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;    //    数据转换参数
  shared_ptr<DataTransformer<Dtype> > data_transformer_;    // 一个data_transformer的指针，用于对数据进行处理。
  bool output_labels_;    // 指明数据中是否包含了lables_, 如果为true则包含。
};


/** @brief 定义了一个Batch的类，它里面只有两个成员变量，Data_和label_, 使用Batch的
    数据结构用于一次预取数据，包含了相应的data和label.
    */
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};


/** @brief 为了加快在获取训练数据时的IO速度，专门定义了一个类，使用多线程完成 数据的预
    先读取到内存中。它继承自:BaseDataLayer类和InternalThread类。其中，InnternalThread类
    是Caffe封装的boost中的Thread类。
  */
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);

  /** @brief 功能描述： BasePrefecthingDataLayer的设置函数， 它首先调用BaseDataLayer中的
      SetUp函数来完成一些通用的动作之后，再对成员变量prefetch_/transformed_data_的初始化等。
    */
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 功能描述： 数据的前向传播时，把当前使用过的prefetch_current放到prefetch_free_队列
      中，然后从prefetch_full_队列中获取一个新的Batch块，把里面的数据装入到top块中cpu数据区。
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 功能描述： 实现的功能与cpu类似，只不过是把数据装载到top中gpu数据区。 */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  /** @brief 功能描述： 该成员函数貌似进行了blob块的读取操作，从prefecth_free中获取空的
      Batch块，然后进行数据的装载，之后放到的pre_fecth_full队列中。
    */
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  vector<shared_ptr<Batch<Dtype> > > prefetch_;      // 一个预读取Batch的vector, 它的大小是由用户在param中的大小决定的。
  BlockingQueue<Batch<Dtype>*> prefetch_free_;       // 空的Batch队列
  BlockingQueue<Batch<Dtype>*> prefetch_full_;       // 满的Batch队列
  Batch<Dtype>* prefetch_current_;                   // 当前使用的Batch的指针，使用完之后会把它push到prefecth_free_中去。

  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
