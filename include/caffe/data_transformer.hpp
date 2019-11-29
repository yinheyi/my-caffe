#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
  @brief 该类实施数据转换，共有四种转换方式： scale操作/裁剪操作/镜像操作/减少mean值操作。
  本次使用到的三种数据存储结构：blob块/datum结构(caffe.proto定义)/Protoblob结构(caffe.proto定义)
  */
template <typename Dtype>
class DataTransformer {
 public:
  /** @brief 在构造函数中做了一些重要的工作。 */
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
    @brief 该函数初始化了成员变量中的随机数生成器的类对象, 用于产生随机整数。在进行裁剪和镜像的
    时候会使用到随机数。
   */
  void InitRand();

  /**
    @brief 数据的转换： Datum =======> Blob<Dtype>*
    @param [in]  datum            要转换的数据, 需要看一下caffe.proto的定义了解它的数据结构。
    @param [out] transformed_blob 转换之后的数据，存放到blob块中, 该blob的num = 1
    */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  /**
    @brief 数据的转换： vector<Datum> =======> Blob<Dtype>*
    @param [in]  datum_vector     要转换的数据
    @param [out] transformed_blob 转换之后的数据，存放到blob块中, 该blob的num = datum_vector的大小
    */
  void Transform(const vector<Datum> & datum_vector, Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
    @brief 把保存在cv::Mat类型中的数据(opencv)进行转换。  cv::Mat数据  =====>   Blob<Dtype>*
    @param [in] cv_img 输入数据
    @param [out] transformed_blob 转换之后的数据， 该blob块的num == 1
    */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

  /**
    @brief 把保存在cv::Mat类型中的多个数据进行转换。  vector<cv::Mat>数据  =====>   Blob<Dtype>*
    @param [in] cv_img 输入数据
    @param [out] transformed_blob 转换之后的数据， 该blob块的num == vector的size.

    该函数多次调用了上面的那个函数来实现。
    */
  void Transform(const vector<cv::Mat> & mat_vector, Blob<Dtype>* transformed_blob);
#endif  // USE_OPENCV

  /**
    @brief 把保存到input_blob块内的数据进行转换。 Blob<Dtype>* =====> Blob<Dtype>*
    @param [in]  input_blob 要转换的输入数据。
    @param [out] transformed_blob 转换之后存放的数据。
    input_blob内可能存在多个图片，使用相同的参数进行转换。 
    */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
    @brief 根据datum内的数据，推测出数据的shape,并返回.
    @param [in] datum 输入的数据块
    @return 返回数据的shape值, 是一个vector.
    */
  vector<int> InferBlobShape(const Datum& datum);

  /**
    @brief 根据vector<datum>内的每一个数据，推测出数据的shape,并返回.
    @param [in] datum 输入的数据块
    @return 返回数据的shape值, 是一个vector.
    */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);

#ifdef USE_OPENCV

  /**
    @brief 根据vector<cv::Mat>内的每一个数据，推测出数据的shape,并返回.
    @param [in] datum 输入的数据块
    @return 返回数据的shape值, 是一个vector.
    */
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);

  /**
    @brief 根据cv::Mat内的数据，推测出数据的shape,并返回.
    @param [in] datum 输入的数据块
    @return 返回数据的shape值, 是一个vector.
    */
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

 protected:

  /**
    @brief 该函数产生一个在区间[0, n)内的整数值。
    @param [in] 上边界(不包含)
    */
  virtual int Rand(int n);

  /** 
    @brief它是完成数据处理与转换的真正干活的函数。 datum数据类型  =====> Dtype* 指针指向的内存.
    @param [in] datum 要转换的数据。
    @param [out] transformed_data 转换之后的数据。
   */
  void Transform(const Datum& datum, Dtype* transformed_data);

  TransformationParameter param_;     // 转换操作的控制参数
  shared_ptr<Caffe::RNG> rng_;        // 随机数生成器的类对象
  Phase phase_;                       // 运行的哪一个阶段，是TRAIN还是TEST
  Blob<Dtype> data_mean_;             // 用于保存从mean文件中读取到的mean值, 每一个元素对应了一个mean值。
  vector<Dtype> mean_values_;         // 用于保存从transformationParameter中取得在mean值.  每一个channel对应一个mean值。
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
