#include <vector>

#include "caffe/layers/input_layer.hpp"

namespace caffe {

/*  从layer_param_中可以得到input_param, 从innput_param中可以得到它设定的每一个top块的shape，
    可能一个shape都没有，也可能只有一个shape(此时所有的top块都reshape成该shape), 也可能有
    num_top(它top块的个数)个shape, 此时每一个shape与一个top块一一对应.
 */
template <typename Dtype>
void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const InputParameter& param = this->layer_param_.input_param();
  const int num_shape = param.shape_size();
  CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }
}

INSTANTIATE_CLASS(InputLayer);
REGISTER_LAYER_CLASS(Input);

}  // namespace caffe
