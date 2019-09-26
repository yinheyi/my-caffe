#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** 该类定义了所有的神经元层的接口，神经元层就是激活函数层了，例如sigmoid函数/tanh函数/relu函数等。
    激活函数层的输入永远与输出的blob块的shape是相同的，一一对应嘛，只是通过激活函数对输入进行了映
    射而已。
 */
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}

  /** @brief 功能描述： 该函数的把top的shape设置为bottom相同的shape.
      @param [in] bottom 神经元层的输入blob块
      @param [in] top    神经元层的输出blob块
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 功能描述: 该函数返回神经元层输入需要的blob块的个数，它只需要一个blob块就可以了。 */
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  /** @brief 功能描述: 该函数返回神经元层输出的blob块的个数，它只输出一个blob块就可以了。*/
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
