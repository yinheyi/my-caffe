#ifndef CAFFE_DROPOUT_LAYER_HPP_
#define CAFFE_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
  @brief 1. dropout层，一种正则化的机制，对每一个神经元的输出以一定的概率置为0,
  也就是丢弃它。  
  2. 该层只有一个重要的参数，dropout_ratio, 表示以多大的概率丢弃一个神经元的输出。
  3. 该类继承自NeuronLayr类。
  */
template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  /** @brief 该函数完成设置本层样本的参数，包含成员变量threshod_和scale_的值。 */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /**
    @brief 本函数首先调用NeuronLayer基类中的reshape函数对top块进行reshape, 然后再对
    成员变量rand_vec_进行reshape. rand_vec_保存了每一个输入是否被dropout, 它内部的值
    为0或1,表示是否被dropout掉，是通过伯努利分布得到的。
    */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Dropout"; }

 protected:
  /** @brief Dropout层的前向传播。*/
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief Dropout层的梯度反向传播。*/
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<unsigned int> rand_vec_;    // 保存了每一个输入是否被dropout的mask值(0或者1)
  Dtype threshold_;                // 保存dropout_ratio值。
  Dtype scale_;                    // 它的值等于 1/（1-dropout_ratio).
  unsigned int uint_thres_;        // 真不知道它是干什么的，代码中也没有用到它。
};

}  // namespace caffe

#endif  // CAFFE_DROPOUT_LAYER_HPP_
