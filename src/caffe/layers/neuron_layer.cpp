#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

// 该函数把top的shape设置为与bottom的shape相同。
template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

/* INSTANT_CLASS的宏定义如下所示：  使用类模板实例化了两个类。
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>
 */
INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
