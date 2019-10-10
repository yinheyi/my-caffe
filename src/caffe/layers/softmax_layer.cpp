#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  // 在该维度上进行softmax操作, 该维度的值就相当于要分为多少类别。
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  // 一维向量中的值全部设置为1, 目的就是使用矩阵相乘函数求累加和。
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);

  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);

  // 初始化scale_的shape, 我发现在下面的代码中，scale_最多使用了inner_num个空间大小啊。
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

// 下面的函数代码，要知道sfotmax层干了什么事才行,要好好的推导一下。
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();

  int channels = bottom[0]->shape(softmax_axis_);  // 它的值就相当于分类的类别数。
  int dim = bottom[0]->count() / outer_num_;       // 注意dim的大小。

  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {

    // 该双层for循环求每一个channel的最大值， 最后求出来的scale_data可以看用是一个
    // 1 * inner_num_的向量。
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }

    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);

    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);

    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);

    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

// 下面求导数的代码， 你需要自己把公式推导一遍才行，具体不说了。
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
