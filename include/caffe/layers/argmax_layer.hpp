#ifndef CAFFE_ARGMAX_LAYER_HPP_
#define CAFFE_ARGMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
  @brief argmax层用于多分类任务中，用于对分类结果的预测。
  例如一共有10类，有一个样本s, s属于每一类的概率分别为p0, p1, .... p9, 选择概率最大的
  那个index作为预测的分类结果。
  */
template <typename Dtype>
class ArgMaxLayer : public Layer<Dtype> {
 public:
  explicit ArgMaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ArgMax"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  bool out_max_val_;      // 是否要输出最大的概率值
  size_t top_k_;          // 输出前k个高的概率值的label或概率值。
  bool has_axis_;        // 是否指定在哪一个轴上进行argmax, 如果没有设置的话，会在除第0轴之外的所有轴上求argmax.例如：
                         // bottom[0]的shape为[a,b,c,d], 如果没有指定axis, 则认为样本数为a, 样本的类别数为b*c*d, 即在b/c/d轴是求argmx.
  int axis_;             // 指定的轴。
};

}  // namespace caffe

#endif  // CAFFE_ARGMAX_LAYER_HPP_
