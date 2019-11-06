#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/argmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ArgMaxParameter& argmax_param = this->layer_param_.argmax_param();

  // 设置out_max_val_,top_k_,以及has_axis_和axis_四个成员变量的值。
  out_max_val_ = argmax_param.out_max_val();
  top_k_ = argmax_param.top_k();
  has_axis_ = argmax_param.has_axis();
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  if (has_axis_) {
    axis_ = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
    CHECK_LE(top_k_, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    CHECK_LE(top_k_, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  if ( num_top_axes < 3 )
      num_top_axes = 3;
  std::vector<int> shape(num_top_axes, 1);    // 初始化为1.

   // 使用第axis_维度存放top_k_个的最大概率值或预测的结果值。
  if (has_axis_) {
    shape = bottom[0]->shape();
    shape[axis_] = top_k_; 
  }
  // 如果没有设置axis_时，shape为[样本数，1, top_k_]或者[样本数，2, top_k_].
  else {
    shape[0] = bottom[0]->shape(0);
    shape[2] = top_k_;
    if (out_max_val_)   // shape[1的默认初始化的值是1
      shape[1] = 2;
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int dim;       // 指定axis上的size大小，也就是总的类别数目。
  int axis_dist; // axis上的值增加1时， 内存布局上跨越了多少。 例如：shape = [a,b,c,d], 如果axis = 1(从0开始计数), 则axis_dist = c*d.
  if (has_axis_) {
    dim = bottom[0]->shape(axis_);
    // Distance between values of axis in blob
    axis_dist = bottom[0]->count(axis_) / dim;
  } else {
    dim = bottom[0]->count(1);
    axis_dist = 1;
  }

  int num = bottom[0]->count() / dim;    // 总的样本数
  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);   // 存放一个样本被划分到第一类别的概率值以及对应的label.
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      // 下面求坐标的公式真长。。就是第i个样本划分到第j个类别上的概率值在bottom_data中的下标。
      // 画个图推推就差不多明白了。
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < top_k_; ++j) {
      // 1. 当指定了输出max_val_并且也设置了axis_时，只输出了了前top_k个概率值。
      // 2. 当指定了输出max_val_, 没有设置了axis_时，输出概率值和label值。
      // 3. 当没有指定输出max_val_时，只输出label值。
      if (out_max_val_) {
        if (has_axis_) {
          // Produces max_val per axis
          top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist] = bottom_data_vector[j].first;
        } else { 
          // Produces max_ind and max_val
          // 此时，top_data的shape = [样本数，2, top_k].
          top_data[i * 2 * top_k_ + j] = bottom_data_vector[j].second;
          top_data[i * 2 * top_k_ + top_k_ + j] = bottom_data_vector[j].first;
        }
      } else {
        // Produces max_ind per axis
        top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
          = bottom_data_vector[j].second;
      }
    }
  }
}

INSTANTIATE_CLASS(ArgMaxLayer);
REGISTER_LAYER_CLASS(ArgMax);

}  // namespace caffe
