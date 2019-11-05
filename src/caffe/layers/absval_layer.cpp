#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // 求输入的绝对值放到输出中
  caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();

  // propagate_down[0]bottom的第0个blob块是否需要计算反向梯度.
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 先计算出输入的正负号
    caffe_cpu_sign(count, bottom_data, bottom_diff);
    // 再乘以loss对输出的梯度值, 把结果存放到bottom_diff中。
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsValLayer);
#endif

INSTANTIATE_CLASS(AbsValLayer);
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
