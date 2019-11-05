#ifndef CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
  @brief 用于多分类任务中求分类结果的正确率。
  */
template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Accuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }      // 输入的blob一定是2个，一个是data，一个是label.

  // 如果输出top块有两个blob块，第一个存放一个标量,全局的正确率;
  // 第二个blob块是一个vector,存放每一类别的blob块。
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // 该层不需要梯度的反向传播，因为loss函数不会使用到该层。
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int label_axis_;           // 在输入的blob块的第几维进行计算正确率, 例如：一个blob块的shape为 C * M * N, 如果label_axis_等于1的话，
                             // 则说明了一个类别的数目等于M,一共对C * N个样本进行分类。
  int outer_num_;
  int inner_num_;

  int top_k_;                // 计算正确率时，只要在输出结果的top-k中存在正确的类别，就认为它是正确的。
                             // 通常情况下，top_k_的值为1, 但是很多时候也会计算top-5的正确率。

  bool has_ignore_label_;    // 是否存在忽略不计的种类。
  int ignore_label_;         // 忽略不计的种类的label.

  Blob<Dtype> nums_buffer_; // 用于保存每一种类别的样本数，它的size大小等于类别数。
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
