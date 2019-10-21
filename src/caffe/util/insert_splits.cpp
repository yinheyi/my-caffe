#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

namespace caffe {

void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  
  // blob块的名字  ----->>  blob块被作为top时所在的<layer的ID, layer的第几个top的ID>的映射。
  map<string, pair<int, int> > blob_name_to_last_top_idx;
  
  // bottom块的<LayerID, layer内第几个bottom的ID> ------>>  该bottom在一上层作为top时的<LayerID, Layer内第几个top的ID>
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
  
  // blob块被作为top时所在的<layer的ID, layer的第几个top的ID> ------->>   该blob块被上层layer作为bottom块的计数值
  map<pair<int, int>, int> top_idx_to_bottom_count;
  
  //  blob块被作为top时所在的<layer的ID, layer的第几个top的ID>    ---->>  该blob块被用于求loss时的相应的权重系数
  map<pair<int, int>, float> top_idx_to_loss_weight;
  
  // top块的ID<layerID, layer中第几个top>  ------>  该top块新生成的split层中的top块的使用的情况的计数，每使用一个就会加1.
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  
  // layer在net中的ID --------->>  layer的名字
  map<int, string> layer_idx_to_layer_name;
  
  // 该for循环遍历每一个net中的每一个layer并收集相关的信息放到上面定义好的map中, 为接下来的param_split服务。
  for (int i = 0; i < param.layer_size(); ++i) {

    // 获取当前layer的名字以及它在net中的位置ID(指的是它是第几个layer)
    const LayerParameter& layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();

    // 该for循环遍历当前layer中的每一个bottom块进行相当操作：
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      // 获取当前blob块的name.
      const string& blob_name = layer_param.bottom(j);
      
      // 除了输入的layer没有bottom块之外，其它的layer都应该存在bottom块的，并且这个bottom块
      // 一定是上一层layer中的top块。在遍历上一层layer的top块时，已经把相关信息添加到了blob_name_to_last_top_idx中，
      // 如果当前layer中的一些bottom块不存在 ，则说明了上一层的layer与当前层的layer连接不起来，
      // 因此必须报错，终止程序！
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      
      // 求得当前blob块在前层作为bottom时的index以及在上一层作为top时的index, 然后建立映射
      // 关系存放到bottom_idx_to_source_top_idx中。
      const pair<int, int>& bottom_idx = make_pair(i, j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
      
      // 增加当前blob块作为top块时，feed给下一层作为bottom块时的数目。
      ++top_idx_to_bottom_count[top_idx];
    }

    // 该for循环遍历所有的top块，建立top的name至top块ID(包含所在layer的ID以及是第几个top块的ID)的映射关系。
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
    }

    // 如果当前的top的blob块用于求loss, 则也要应该把当前的top块看作是loss层中的bottom块来处理。
    // 因此，下面的for循环遍历所有用于求loss的top块，然后保存top块的Index到loss_weight的映射，同时，如果loss_weight
    // 不为0的话，也应该递增当前top块在下一层中作为bottom块的计数值。
    const int last_loss =
        std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const string& blob_name = layer_param.top(j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      if (top_idx_to_loss_weight[top_idx]) {
        ++top_idx_to_bottom_count[top_idx];
      }
    }
  }

  // 复制输入的net的参数对接下来要构建的param_split进行初始化并clear掉了它里面的layer层参数。
  param_split->CopyFrom(param);
  param_split->clear_layer();

  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layer();
    layer_param->CopyFrom(param.layer(i));

    // 该for循环负责把当前layer的bottom块替换为在上一层中已经对top块进行了split层的top块。
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      const pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];

      // 只有当split_count的个数>1时，才说明了上一层的top块进行了split层生成
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string& layer_name = layer_idx_to_layer_name[top_idx.first];
        const string& blob_name = layer_param->bottom(j);

        // top_idx_to_bottom_split_idx[top_idx]中的值表示下一个可以使用的split层中的top块, 所以当前的
        // bottom使用了一个top块之后，就对它进行了加1.
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
      }
    }

    // 对于当前layer层中的每一个top块，只要它被作为其它层的bottom的次数大于了1,就在该top块
    // 之上生成一个split层, 相当于对该top块进行了分裂复制. 新生成的split层的bottom块数目为1,
    // 就是当前的top块, 而split层的top块的数目等于被其它层使用的次数。
    for (int j = 0; j < layer_param->top_size(); ++j) {

      // 求得top块被其它layer使用的次数，也是接下来要新生成的split层中的top块的个数。
      const pair<int, int>& top_idx = make_pair(i, j);
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {

        // 求得需要的参数值，并配置新增加的split层的param.
        const string& layer_name = layer_idx_to_layer_name[i];
        const string& blob_name = layer_param->top(j);
        LayerParameter* split_layer_param = param_split->add_layer();
        const float loss_weight = top_idx_to_loss_weight[top_idx];
        ConfigureSplitLayer(layer_name, blob_name, j, split_count, loss_weight, split_layer_param);

        // 如果loss_weight不为0, 说明在求loss时会使用到该top块，而上层使用的不会再是该top块，而是该top块
        // 分裂之后新生成的split层的中的第一个top块。所以这里把当前层的weight给清空了。
        // 至于top_idx_to_bottom_split_idx吧，它作为一个动态的增加，每当使用一个split层中的top块，就增加1.
        // 这里求loss层会使用一个split层中的top块，所以加1了。 
        if (loss_weight) {
          layer_param->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }
}

// 该函数配置新生成的split层，包括配置它的bottom块，它的多个top块.
void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
    if (loss_weight) {

      // 这里要注意，它只能split层中的第一个top块中的loss_weight置1了，其它的默认是0.
      // 因为第一个top块是用于计算loss的
      if (k == 0) {
        split_layer_param->add_loss_weight(loss_weight);
      } else {
        split_layer_param->add_loss_weight(0);
      }
    }
  }
}

// 生成split层的名字 
string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}

// 生成split层中的每一个top块的名字。
string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx) {
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
