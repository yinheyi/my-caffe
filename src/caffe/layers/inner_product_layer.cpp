#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // 从参数中分别读取到： 输出中的神经元个数/ 是否使用偏置参数/ 是否对权值矩阵进行转置
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;

  /* axis表示输入的blob块中，从哪一维度开始之后的全部数值作为输入,  例如blob块的shape
     为[6, 3, 4, 3],并且axis值为1，则3 * 4* 3的数据作为全连接层的输入，6表示有6个这样
     的输入，相当于batch_size。 求出的K_值正好是输入中的神经元个数 */
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  K_ = bottom[0]->count(axis);

  /* blobs_是在基类中定义的成员变量，它是一个vector, blobs_[0]用于存放权值, blobs_[1]用
     于存放偏置。*/
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    /* 设置权值矩阵的shape，并且调用weight_filler函数对权值进行初始化。具体调用哪一个
     权值初始化函数，由layer_param_中的参数决定。 */
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // 初始化偏置值。
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  
  /* param_propagate_down是一个vector,它的size()与blobs_的size()是相同的， 在blobs_中的
     blobs_[0]表示权值参数，blobs[1]表示偏置参数， param_propagate_down[0]表示是当误差返
     回传播时，是否要计算权值的导数， param_propagate_down[1]表示是否要计算偏置的导数。*/
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  /* axis表示输入的blob块中，从哪一维度开始之后的全部数值作为输入,  例如blob块的shape
     为[6, 3, 4, 3],并且axis值为1，则3 * 4* 3的数据作为全连接层的输入，6表示有6个这样
     的输入，相当于batch_size。 求出的new_K值正好是输入中的神经元个数, M_相当于batch_size
     的大小. */
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  M_ = bottom[0]->count(0, axis);

  /** 下面求了对应的top_shape的值。很有意思，举例说明：如果输入的shape为[4, 5, 6, 7, 8],而
      axis的值为3,则相当于把 7 * 8的数据拉伸到一维数据作为一次训练的输入K_，输出的神经元个
      数为N_, 权值矩阵为K_ * N_, 因此输出的shape为[4, 5, 6, N_]. */
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  // 这里设置了每一个偏置值的乘法因子, 即每一个bitch_size的偏置值的 multiplier值。 
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}


/* 调用gemm()函数来完成矩阵运算, 参数如下：
   Caffe_gemm(矩阵A是否转置， 矩阵B是否转置， P(矩阵A)的行， P(矩阵B)的列，
             P(矩阵A)的列或P(矩阵B)的行，Alpha, 矩阵A的地址， 矩阵B的地址，
             Beta, 矩阵C).
   矩阵C = Alpha * 矩阵A × 矩阵B + Beta * 矩阵C

    如果矩阵A不进行转置，则P(矩阵A）等于矩阵A, 否则等于矩阵A的转置。
    如果矩阵B不进行转置，则P(矩阵B）等于矩阵B, 否则等于矩阵B的转置。

    例如：
    1. 矩阵A为M×N，矩阵B为N×P, 则：
    gemm(不转置，不转置，M,P,N,Alpha,矩阵A的地址，tdblB的地址，Beta,矩阵C的地址)
    2. 矩阵A为M×N，矩阵B为P×N, 则：
    gemm(不转置，转置，M,P,N,Alpha,矩阵A的地址，tdblB的地址，Beta,矩阵C的地址).
    
    通过例1和例2看到了吧，转置由什么决定的,由它实际的行与列，与参数中的无关！！！
   */
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  /* 其中: bottom_data为 M_ × K_, 其中M_相当于样本数， K_为输入中的神经元个数。
           top_data 为 M × N_,        N_为输出中的神经元个数。
           weight_为K_ × N_(transpose为true)或N_ × K_(transpose为false).
   */
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  /* 如果使用了偏置参数，则加上相应的值。 */
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  /* 计算对权值的导数。当trannspose_为true时，权值矩阵为 K_ × N_;
                       当transpose_为false时，权值矩阵为N_ × K_;
     top_diff是一个 M_ × N_的矩阵;
     bottom_data是一个 M_ × K_ 的矩阵;
     当transpose_为true时，diff_of_weight矩阵 = bottom_data矩阵的转置 × top_diff矩阵。
     当transpose_为false时，diff_of_weight矩阵 = top_diff矩阵的转置 × bottom_data的矩阵;

     最后新得到权值的导数是多个单独的样本的求得的结果的和。
   */
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }

  /* 计算偏置的导数, 这里使用了gemv()函数。
     gemv(矩阵A是否转置，矩阵A的行，矩阵A的列, alpha, 矩阵A的地址，
          向量B的地址， Beta, 矩阵B的地址)
     top_diff是一个 M_ × N_的矩阵;
     bias_multiplier是一个size为M的一维向量.
     偏转的导数是一个N × 1的矩阵
   */
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }

  /* 梯度继续反向传播，求对输入数据的导数。
     top_diff是一个 M_ × N_的矩阵;
    当trannspose_为true时，权值矩阵为 K_ × N_;
    当transpose_为false时，权值矩阵为N_ × K_;
    bottom_data是一个 M_ × K_的矩阵

    输入的导数矩阵为：top_diff矩阵 × 权值矩阵的转置(transpose_为true)
                      top_diff矩阵 × 权值矩阵(transpose_为false)
   */
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
