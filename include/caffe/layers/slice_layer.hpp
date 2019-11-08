#ifndef CAFFE_SLICE_LAYER_HPP_
#define CAFFE_SLICE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
  @brief sliceLayer层的作用是对blob块进行切片操作。 该层两个重要的参数是
  slice_axis和 slice_point(是一个vetor)，分别指明了沿哪一个轴进行切片以及
  在那个轴上切片的位置。
  */
template <typename Dtype>
class SliceLayer : public Layer<Dtype> {
 public:
  explicit SliceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  /** 
    @brief 虚函数，建立slice层，它的作用是初始化了成员变量slice_axis_
    和slice_point_的值。
    */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
    @brief reshape函数，主要对top块进行reshppe. bottom块的size一定是1,而
    top块最少是一个，也可能是多个.当设置了slice_point_时，top块的个数应该
    等于slice_point_.size() + 1 (至于为什么加1, 就比如一根绳子剪n刀会变成
    n+1段相同。)
    */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Slice"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  /** @brief 前向传播函数。*/
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /** @brief 梯度的反向传播。*/
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  // 假设bottom[0]的shape = [a,b,c,d,e,f], 并且在c轴(对应slice_axis_=2）进行切片，
  //则下面的成员变量的值分别对应了：
  int count_;                   // 它的值等于a*b*c*d*e*f，表示共有多少输入元素。
  int num_slices_;              // 它的值等于a*b
  int slice_size_;              // 它的值等于d*e*f.
  int slice_axis_;              // 它的值等于2.
  vector<int> slice_point_;    // 在给定轴上切片的位置点，例如轴长为6, slice_point_的值
                               // 为[1,4,5],则切片的结果分别为：[0,1), [1,4), [4,5), [5,6),
                               // 总计4份, 相应的top块的数目也应该是4.
};

}  // namespace caffe

#endif  // CAFFE_SLICE_LAYER_HPP_
