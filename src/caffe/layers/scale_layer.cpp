#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();

  // 如果bottom块的数目为1，说明了没有指定的scale块的值(因为bottom[1]可以用于指定为scale的值),
  // 那么,scale的值一定是作为了本层的参数了，它存放了blobs_块中。如果blobs_的size()>0，说明已经
  // 初始化了scale的值。  但问题是：谁会初始化该值呢？？？
  if (bottom.size() == 1 && this->blobs_.size() > 0)
  {
      LOG(INFO) << "Skipping parameter initialization";
  }
  else if (bottom.size() == 1)  // 此时说明没有初始化scale的值就需要对它进行初始化
  {
    // 首先从参数中获取axis_和num_aexs的值. 
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }

    // 接着，通过axis_和num_aexs的值来确定scale块的shape.
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);  // 注意这里处理num_axes=-1的情况。
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(scale_shape));

    // 接着，对scale块进行初始化，如果没有指定初始化方法的话，会初始化为1.
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }

  // 如果在参数中选择了增加bias值，则创建一个bias层，用于完成 + bias 的操作。
  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } else {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    bias_bottom_vec_.resize(1);
    bias_bottom_vec_[0] = bottom[0];
    bias_layer_->SetUp(bias_bottom_vec_, top);
    if (this->blobs_.size() + bottom.size() < 3) {
      // case: blobs.size == 1 && bottom.size == 1
      // or blobs.size == 0 && bottom.size == 2
      bias_param_id_ = this->blobs_.size();
      this->blobs_.resize(bias_param_id_ + 1);
      this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
    } else {
      // bias param already initialized
      bias_param_id_ = this->blobs_.size() - 1;
      bias_layer_->blobs()[0] = this->blobs_[bias_param_id_];
    }
    bias_propagate_down_.resize(1, false);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)

  // 当num_axes的值为0时，表示scale值是一个标量，相当于所有的输入值都使用一个scale值进行缩放操作。
  // 此时，把axis设置为0了，这样的接下来的计算很高效。
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] == top[0]) {  // in-place computation
    temp_.ReshapeLike(*bottom[0]);
  } else {
    top[0]->ReshapeLike(*bottom[0]);
  }

  // sum_result_向量值用于保存在inner_dim上求和之后的值, 求和之后就是一个大小就
  // 等于outer_dim * scale_dim.
  sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
  // sum_multiplier用于通过矩阵和向量的乘积实现对矩阵行求和，它里面的值全为1,至于
  // 它的size()大小选择outer_dim和inner_dim中最大值的原因是在求scale的梯度时需要在
  // inner_dim和outer_dim上对梯度的内积矩阵按行求和的。
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  // 这里有一点不明白，为什么只需要判断最后一个值是不是等于1就决定了整个sum_multiplier的
  // 值是不是1呢？
  if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)) {
    caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
  if (bias_layer_) {
    bias_bottom_vec_[0] = top[0];
    bias_layer_->Reshape(bias_bottom_vec_, top);
  }
}

// 正向传播。
template <typename Dtype>
void ScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // 如果是原位计算的话，这里把输入的值在被重写之前保存一份，在梯度的反向传播时会使用到。
  if (bottom[0] == top[0]) {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
               temp_.mutable_cpu_data());
  }
  const Dtype* scale_data = ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const Dtype factor = scale_data[d];
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
  if (bias_layer_) {
    bias_layer_->Forward(bias_bottom_vec_, top);
  }
}

// 反向传播
template <typename Dtype>
void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];

  // 求scale值的梯度
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->cpu_data();

    // 下面这里其实没有必要搞这么复杂的，真的很影响代码的可读性。下面的代码为了效率的考虑，
    // 处理了特殊的场景： 
    // 1. 每一个输入都有一个单独scale值的情况，此时scale->count() == bottom[0]->count().
    // 2. inner_dim等于1的场景
    // 3. outer_dim等于1或者outer_dim * sclale_dim =1的场景
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.

    const bool is_eltwise = (bottom[0]->count() == scale->count());
    Dtype* product = (is_eltwise ? scale->mutable_cpu_diff() :
        (in_place ? temp_.mutable_cpu_data() : bottom[0]->mutable_cpu_diff()));
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Dtype* sum_result = NULL;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (sum_result_.count() == 1) {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (scale_param) {
          Dtype result = caffe_cpu_dot(inner_dim_, product, sum_mult);
          *scale_diff += result;
        } else {
          *scale_diff = caffe_cpu_dot(inner_dim_, product, sum_mult);
        }
      } else {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_cpu_diff() : sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }

      if (outer_dim_ != 1) {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (scale_dim_ == 1) {
          if (scale_param) {
            Dtype result = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
            *scale_diff += result;
          } else {
            *scale_diff = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
          }
        } else {
          caffe_cpu_gemv(CblasTrans, outer_dim_, scale_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                         scale_diff);
        }
      }
    }
  }

  // 求bottom[0]的梯度
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scale_data = scale->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < outer_dim_; ++n) {
      for (int d = 0; d < scale_dim_; ++d) {
        const Dtype factor = scale_data[d];
        caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);
        bottom_diff += inner_dim_;
        top_diff += inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleLayer);
#endif

INSTANTIATE_CLASS(ScaleLayer);
REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
