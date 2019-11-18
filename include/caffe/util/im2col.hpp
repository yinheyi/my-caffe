#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
/**
  @brief 该函数实现image to column的功能。 
  @param [in] data_im 输入
  @param [out] data_col 输出
  @param [in] pad_h,pad_w  在输入的矩阵四周加了一个为0的边廓, 这两个参数分别表示在高与宽上增加的边廓的宽度。
  @param [in] dilation_h,dilation_w 对kernel的dilation算法(百度)：kernel_dilation = dilation * (kernel - 1) + 1;

  1. dilation的含义:就是在卷积核中间填充0,每列和第行中间填充dilation-1个0.
  2. 具体什么是im2col呢？ 就是给你一个M*N的二维矩阵， 然后呢，一个小的Kernel矩阵在大矩阵上进行移动，
  每一移一次，就把kernel矩阵对应的大矩阵上的元素转换为一个列， 通过一系列的移动，就生成了一个新
  的矩阵了。举个例子：
               |1,   2,  3,  4|
   原矩阵  X = |5,   6,  7,  8|, Kernel矩阵为2*2,移动步长为1, 则进行im2col之后的新矩阵为：
               |9,  10, 11, 12| 
               |13, 14, 15, 16| 

              | 1, 2, 3, 5,  6,  7,  9, 10, 11 |
  结果矩阵 Y =| 2, 3, 4, 6,  7,  8, 10, 11, 12 |
              | 5, 6, 7, 9, 10, 11, 13, 14, 15 |
              | 6, 7, 8, 10,11, 12, 14, 15, 16 |
  */
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col);

/**
  @brief 多维矩阵下的image转换为column的操作。
  多维指是的image是多维的，相应的kernel也是多维的, 相应的转换之后的col也是多维的，
  它们的维度数目是相同的。 
  @param [in] data_im image的数据。
  @parama[in] num_spatial_aexs image或kernel或col的维度数目。
  @param [in] im_shape image的shape, 它的size = num_spatial + 1, 其中im_shape[0]表示image的个数。
  @param [in] col_shape 转换之后的column的shape, 它的size = num_spatial + 1, 其中col_shape[0]表示column的个数。
  @param [in] kernel_shape kernel的shape,它的size = num_spatial_aexs.
  @param [in] pad 在每一个维度上进行pad的值，它的size = num_spatial_aexs.
  @param [in] stride 在每一个维度上的步长值，它的size = num_spatial_aexs.
  @param [in] dilation 在每一个维度上进行dilation的值，它的size = num_spatial_aexs.
  @param [out] data_col 保存输出之后的值。
  */
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col);


/** @brief 二维下的逆转换.  */
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

/** @brief 多维下的逆转换.  */
template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im);

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
    const int col_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
