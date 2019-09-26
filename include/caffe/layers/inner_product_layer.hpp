#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** @brief 该类定义了全连接层。
    在全连接层中， 输入有K_个神经节点， 输出有N_个神经结点, 输入用一个 M_ × K_的矩阵， 
    权值用一个 K_× N_ 的矩阵, 偏置用一个 1×N_ 的一维向量， 输出用一个M_ × N_的矩阵表
    示。 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  /** @brief 功能描述：该函数只使用到了bottom的值，它做了这样的事：从bottom中可以得到输入
      在shape, 该shape结合param中的axis()的值就可以知道输入层的神经元个数，从param中还可
      以得到输出层的神经元个数，如此一来就知道了权值和偏置的形状，因此对它们进行初始化。*/
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 功能描述：该函数干了两件事：1. 设置top数据中的shape. 2. 设置bias_multiplier
      的shape并初始化为1.  */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 功能描述：返回当前layer的名字。*/
  virtual inline const char* type() const { return "InnerProduct"; }
  
  /** @brief 返回全连接层的输入需要多少个数据块， 它只需要一个blob块就够了。 */
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  /** @brief 返回全连接层的输出需要多少个数据块， 输出也只需要一个blob块就够了。 */
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /** @brief 在cpu上计算全连接层前向传播的函数， 其实就是输入与权值的矩阵相乘, 如果存在
      偏置，则把矩阵相乘的结果的基础上增加偏置矩阵M_ * N_ 就可以了。矩阵相乘时，使用
      gemm()来完成。*/
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /** @brief 在gpu上计算全连接层前向传播的函数  */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;    // M _的含义相当于多少个训练样本
  int K_;    // K_表示输入中有多少个神经节点
  int N_;    // N_表示输出中有多少个神经节点
  bool bias_term_;    // 表示是否使用偏置值。
  Blob<Dtype> bias_multiplier_;    // 每一个训练样本的偏置的乘数因子, 它的size()是M_.
                                   // 意思就是可以控制不同的样本偏置值的因子大小。
  bool transpose_;  // 表示权值矩阵是否进行转置, 当为true时，权值矩阵的shape为 K_ × N_, 当为false时， 权值矩阵的shape为 N_ × K_.
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
