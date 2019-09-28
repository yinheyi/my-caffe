#ifndef CAFFE_INPUT_LAYER_HPP_
#define CAFFE_INPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net by assigning tops directly.
 *
 * This data layer is a container that merely holds the data assigned to it;
 * forward, backward, and reshape are all no-ops.
 */
template <typename Dtype>
class InputLayer : public Layer<Dtype> {
 public:
  explicit InputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  /** @brief 功能描述： 该函数作的唯一工作就是把top块reshape成input_param参数
      中指定的shape.
      @param [in] bottom 没有使用到。
      @param [in] top    存放数据的blob块。
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "Input"; }

  // input 层作为最开始的一层，不需要bottom blob吗? 那它的数据是从哪里来的？
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  // 如果有label的话，可能输入是2个blobs块吧。
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe

#endif  // CAFFE_INPUT_LAYER_HPP_
