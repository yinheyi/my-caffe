#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Caffe, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BatchNormLayer` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
 public:
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
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

  Blob<Dtype> mean_;                     // 均值，一个size= channel的向量
  Blob<Dtype> variance_;                 // 方差，一个size= channel的向量
  Blob<Dtype> temp_;                     // 它的shape与bottom[0]相同, 用于保存中间计算过程中的临时变量(标准差broadcast成bottom[0]的shape的值)
  Blob<Dtype> x_norm_;                   // 它的shape与bottom[0]相同, 保存了一份标准化之后的数据到它里面（与top块的值完全相同）,之所以备份一份，
                                         // 是因为在backward()时需要top[0]的值，但是呢，不能保证后面的layer层会不会执行in-place计算，这样会把top[0]
                                         // 的值覆盖了。


  bool use_global_stats_;                // 表示在进行标准化时，是否使用已经保存的均值与方差，在测试阶段应该置为true.
  Dtype moving_average_fraction_;        // 求累加和时，对旧值乘以该比例，即New = Old * 该值 + current.
  int channels_;                         // channel的数目，对每一个channel上的元素(再加上batch_size)进行求均值和方差
                                         // 所以，channel的数目等于均值或方差的数目。
  Dtype eps_;                            // 

  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob<Dtype> batch_sum_multiplier_;      // 对batch维度求和时，使用该向量来对矩阵的行求和。
  Blob<Dtype> num_by_chans_;              // 它用于存放对blob块在spatial维度(也就是一个channle上)求和之后的结果.
  Blob<Dtype> spatial_sum_multiplier_;    // 对spatial维度求和时，使用该向量来对矩阵的行求和。 
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
