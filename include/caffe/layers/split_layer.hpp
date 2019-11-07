#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
  @brief split层，它实现的功能：对输入进行分裂复制多份到输出中。  
  1. 在正向传播中，不需要把输入拷贝每一份到输出中，输出块直接share输入的blob块即可，也就是
  仅仅复制指针就OK了, 只存在一份data块。
  2. 在梯度的反向传播中，每一个输出块都有自己独立的diff块. 所有输出的diff块按元素进行相加之
  后，就得到了输入的diff值。
  */
template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  /** @brief 该函数对输出的每一个top块进行reshape成输入的bottom块的大小。*/
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Split"; }

  /** @brief 输入的blob块一定是1个。 */
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  /** @brief 输出的blob块最少是一个，也可能是多个。 */
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  // 前向传播操作
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // 反向传播, 其中propagate_down表示每一个bottom块是否需要求梯度。
  // bottom_diff = top[1]_diff + top[1]_diff + ... + top[n]_diff.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int count_;      // 输入块中元素的个数，即bottom[0].count()的值。
};

}  // namespace caffe

#endif  // CAFFE_SPLIT_LAYER_HPP_
