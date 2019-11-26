#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();

  // 使用全局的pooling时，不需要指定kernel_size,因为它的整个w*h上，进行pooling,最后得到一个输出。
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }

  // 检测pad参数
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";

  // 检测stride参数。
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";

  global_pooling_ = pool_param.global_pooling();

  // round_mode的参数控制着当pooling的输出不正好取整时，是向上取整还是向下取整. 即floor() 还是ceil().
  round_mode_ = pool_param.round_mode();

  // 全局pooling时，注意kernel_h和kernel_w的值的大小。
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }

  // global_poling时的pad和stride值。
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }

  // pad非0的限制
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // pooling 只支持4维的？？
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, " << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // 在setup函数中已经对它进行初始化过了, 这里为什么又写一遍?
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }

  // 计算pooling之后的输出的大小。
  switch (round_mode_) {
  case PoolingParameter_RoundMode_CEIL:
    pooled_height_ = static_cast<int>(ceil(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(ceil(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    break;
  case PoolingParameter_RoundMode_FLOOR:
    pooled_height_ = static_cast<int>(floor(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(floor(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    break;
  default:
    LOG(FATAL) << "Unknown rounding mode.";
  }

  /* 在pad时，确保了w和h中的最后一个pooling区域还是pad出来的。 但是，为什么不处理
     第一个pooling区域呢？ 还有，万一pad的值是kernel的w或h的多倍时，也不能确保pooling
     区域不是pad出来的呢？  */
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.

    // (pooled_height_ -1) * stride_h 就是最后一个pool区间在image上开始的index.(第1个是0,第2个是stride_h, 第3个是2 * stride_h,  ...).
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }

  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }

  // If max pooling, we will initialize the vector index part.
  // max_idx_指明了pooling的值是pooling区域中的哪一个元素.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }

  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}


// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
// 正向传播。
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  // We'll output the mask to top[1] if it's of size >1.
  // 这个mask就是就是pooling后的值来自pooling区域的哪一个元素。 当top块大于1时，把这个信息保存到
  // top[1]中，否则保存到max_idx_或rand_idx_中。
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_mask = NULL;

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {

        // 下面两个for循环求每一个pooling后的值:
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

            // pooling区域的起始位置和终止位置。
            // 有一个疑问: 当pad_h或pad_w很大时，会出现start = 0, end < 0的情况, 此时就不会执行接下来的两层for循环。
            // 对应的mask[index]的值还是初始化的-1, 在反向传播上，使用了-1的索引值，软件应该是崩溃的吧。。。
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);

            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }

        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

            // 求pool区域，以及pool区域内的元素个数。
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

// 梯度反向传播
template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

              // 这里是核心
            const int index = ph * pooled_width_ + pw;
            const int bottom_index = use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {

                // top块的梯度值乘以一个系数。因为正向传播中求的是平均值。
                bottom_diff[h * width_ + w] += top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
