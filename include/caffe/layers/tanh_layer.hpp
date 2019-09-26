#ifndef CAFFE_TANH_LAYER_HPP_
#define CAFFE_TANH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/** @brief 该类定义了tanh激活函数层。

    tanh函数的公式定义，以及它的导数如下所示：
    tanh(x) = (e^x - e^-x) / (e^x + e^-x) = (e^2x - 1) / (e^2x + 1)
    tanh'(x) = 1 - tanh(x) ^ 2
  */
template <typename Dtype>
class TanHLayer : public NeuronLayer<Dtype> {
 public:
  explicit TanHLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "TanH"; }

 protected:
  /** @brief 神经网络的前身传播过程, 给定激活函数一个输入值，计算输出值。
      @param [in]  bottom 输入数据的blob块。
      @param [out] top    存放计算得到的输出值。
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 使用gpu计算激活函数值。*/
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 神经网络的梯度反向传播过程, 给定代价函数对激活函数输出值的导数，
       计算对输入值的导数.  */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /** @brief 使用gpu计算输入的梯度值。*/
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_TANH_LAYER_HPP_
