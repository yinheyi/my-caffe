#ifndef CAFFE_SOFTMAX_LAYER_HPP_
#define CAFFE_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief 该类定义了softmax层，该层就是一个softmax函数。输入与输出的shape是相同的。
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Softmax"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }    // bottom的输入只需要一个Blob就OK.
  virtual inline int ExactNumTopBlobs() const { return 1; }       // top的输出也只需要一个blob就OK.

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int outer_num_;        // 这个值相当于输入了多少个图片。
  int inner_num_;        // 这个值相当于把一个图片分别了好几块，每一块都进行softmax,在实际中，大多数下，inner_num_就是1而已，一个图片是一个类别.
  int softmax_axis_;     // 它值表示在第几维度进行softmax操作，例如一个shape为[32,100, 5, 5]，而softmax_axis_为1,则表示在100的那个维度进行softmax,
                         // 相当于共有100类别，此时out_num_的值为32, inner_num_的值为25.

  // 定义这个一维向量(它的大小为shape[softmax_axis_]），目的是为了利用矩阵乘法求累加和,在forward和backward中会使用到。
  Blob<Dtype> sum_multiplier_;

  /// scale is an intermediate Blob to hold temporary results.
  Blob<Dtype> scale_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
