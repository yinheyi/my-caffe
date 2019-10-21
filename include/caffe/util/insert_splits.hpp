#ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
#define _CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
   @brief 该函数对原net的param中进行共享的blob块进行了增加了split层，即进行了分裂复制，
   使得上一层的layers都有它们各自的bottom块。
   @param [in]  param       原来的网格的参数。
   @param [out] param_split 进行了split之后的网络的参数。
   @return 返回值为空。
   */
void InsertSplits(const NetParameter& param, NetParameter* param_split);

/**
  @brief 配置新增加的split layer层的参数。
  @param [in] layer_name 要进行split的top块所归属的layer的名字。
  @param [in] blob_name  要进行split的top块的名字.
  @param [in] blob_idx   要进行split的top块是归属的layer的第几个top块。
  @param [in] blob_idx   要进行split的top块被其它层作为bottom块使用了多少次，也就是split层要split出多少个新的top块。
  @param [in] loss_weight 当前top块中的loss_weight的值。
  @param [out] split_layer_param 要配置的layer的参数。
  */
void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param);

/**
  @brief 构造新生成的split层的名字. 
  @param [in] layer_name 要进行split的top块所归属的layer的名字。
  @param [in] blob_name  要进行split的top块的名字.
  @param [in] blob_idx   要进行split的top块是归属的layer的第几个top块。
  */
string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx);

/**
  @brief 构造新生成的split层第split_idx个的top块的名字。
  @param [in] layer_name 要进行split的top块所归属的layer的名字。
  @param [in] blob_name  要进行split的top块的名字.
  @param [in] blob_idx   要进行split的top块是归属的layer的第几个top块。
  @param [in] split_idx  表示构造的第几个top块的名字。
  */
string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
