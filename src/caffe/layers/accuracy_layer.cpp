#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();    // top_k的意思就是预测结果的前top_k个中有正确的标签就可以了.

  // 是否要不计入正确率的label,有的话，找出来。
  has_ignore_label_ = this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) 
      ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // bottom[1]的count等于总的要分类的样本数, 因为bottom[1]中存放的是每一个样本的label.
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";

  // CanonicalAxisIndex函数对index进行了一个正则化，例如-1正则化到正整数。
  label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  // top[0]是一个标量，里面存放的是accuracy, 即全局的识别率。
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);

  // top[1]是一个向量，它里面存放了每一类别的识别率.
  if (top.size() > 1) {
    vector<int> top_shape_per_class(1, bottom[0]->shape(label_axis_));
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);    // 种类的个数, 0, 1, 2,....., num_labels - 1.

  // 如果要计算每一个类别的识别率，则初始化需要使用到的mums_buffer_(存放每一类别的样本数)
  // 和top[1](存放每一类别的正确识别率).
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }

  int count = 0;    // 用于计数总的样本数(排除了要忽略label的样本)
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {

      // i * inner_num + j 表示第几个样本 , 下面求出第n个样本的label值,label值是从0开始计数的。
      const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_)
        continue;

      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      if (top.size() > 1)
          ++nums_buffer_.mutable_cpu_data()[label_value];

      // 第n样本被识别为正确类别的概率。
      const Dtype prob_of_true_class = bottom_data[i * dim + label_value * inner_num_ + j];
      // 该变量用于统计比正确类别的概率还高的错误类别的数目
      int num_better_predictions = -1;  // true_class also counts as "better" , 所以初始化为-1了。
      for (int k = 0; k < num_labels && num_better_predictions < top_k_; ++k) {
        num_better_predictions +=
          (bottom_data[i * dim + k * inner_num_ + j] >= prob_of_true_class);
      }

      // check if there are less than top_k_ predictions
      if (num_better_predictions < top_k_) {
        ++accuracy;
        if (top.size() > 1)
            ++top[1]->mutable_cpu_data()[label_value];
      }
      ++count;
    }
  }

  // 除以总数， 计算正确识别率。
  top[0]->mutable_cpu_data()[0] = (count == 0) ? 0 : (accuracy / count);
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AccuracyLayer);
#endif

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
