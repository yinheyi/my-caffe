#ifndef CAFFE_SCALE_LAYER_HPP_
#define CAFFE_SCALE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe {

/**
 * @brief Computes the elementwise product of two input Blobs, with the shape of
 *        the latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product. Note: for efficiency and convenience, this layer can
 *        additionally perform a "broadcast" sum too when `bias_term: true`
 *        is set.
 *
 * The latter, scale input may be omitted, in which case it's learned as
 * parameter of the layer (as is the bias, if it is included).
 */
template <typename Dtype>
class ScaleLayer: public Layer<Dtype> {
 public:
  explicit ScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scale"; }

  /**
    @brief 获取scale层的输入块的数目。 当输入为2时，bottom[1]用于存在scale的值;
    如果输入为1时，scale的值作为本层的学习参数，它会根据梯度值进行更新的。
    */
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Layer<Dtype> > bias_layer_;      // 当参数中选择了bias时，会在scale的基础上增加一个bias的值，通过bias_layer来完成。
  vector<Blob<Dtype>*> bias_bottom_vec_;      // bias_layer层的输入
  vector<bool> bias_propagate_down_;          // 表示bias_layer层中的bottom块是否需要梯度的反向传播。
  int bias_param_id_;                         // bias层的偏置值在scale层中的bobs_中的第几个位置上。

  Blob<Dtype> sum_multiplier_;    // 一维向量，里面的值全为1,用于矩阵与向量的相乘实现对矩阵按行求和。
  Blob<Dtype> sum_result_;        // 在求scale的梯度时使用，用于保存product在inner_dim维度上求和之后的结果
  Blob<Dtype> temp_;              // 如果对输入进行原位操作时，作于保存原始的输入的值，在反向传播求梯度时需要。
  int axis_;
  int outer_dim_, scale_dim_, inner_dim_;
};


}  // namespace caffe

#endif  // CAFFE_SCALE_LAYER_HPP_
