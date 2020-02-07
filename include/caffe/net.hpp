#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
 /**
   @brief net的构造函数，所有的工作交给Init()函数来完成。
   */
  explicit Net(const NetParameter& param);
  
 /**
   @brief net的构造函数，首先从param_file路径中读取net的参数NetParameter中，然后再使用函数参数中的
   phase/level/stages来设置NetParamter中的参数，最后还是把工作交给了Init()函数来完成。 
   */
  explicit Net(const string& param_file, Phase phase, const int level = 0, const vector<string>* stages = NULL);
  virtual ~Net() {}

  /**
    @brief 使用网格参数初始化Net. 这里面干了好多事情:  
    1. 过滤掉与当前net的状态信息不匹配的layers.  
    2. 对一个layer的top块被多个layer共享的情况，进行了插入split层layer的操作。  
    3. 
    */
  void Init(const NetParameter& param);

  /**
    @brief 网络的正向传播函数，它调用ForwardFromTo()函数来完成实际的工作。
    @param [out] loss 它是一个指针，用于传出计算到的loss值。
    @return 返回是的输出的blobs.
    */ const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);

  /**
    @brief 网络的前向传播函数, 计算[start, end]的layers。(左闭右闭格式)
    @param [in] start 开始的layer的id(从0开始计数).
    @param [in] end 终止的layer的id

    The From and To variants of Forward and Backward operate on the
    (topological) ordering by which the net is specified. For general DAG
    networks, note that (1) computing from one layer to another might entail
    extra computation on unrelated branches, and (2) computation starting in
    the middle may be incorrect if all of the layers of a fan-in are not
    included.
    */
  Dtype ForwardFromTo(int start, int end);

  /** @brief 网络的前向传播函数, 从start到最后一层layer.  */
  Dtype ForwardFrom(int start);

  /** @brief 网络的前向传播函数, 从第0层到end层的layer.  */
  Dtype ForwardTo(int end);

  /** @brief 在网格反向传播之前，把所有权值的梯度都置为0. */
  void ClearParamDiffs();

  /** @brief 进行网格的反向传播，计算相应的权值梯度。 */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief 遍历所有的layer进行reshape操作。
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();

  /**
    @brief Shares weight data of owner blobs with shared blobs. 它需要使用到
    成员变量 params_和param_owners_.

    @details 对于共享权值的blob块，这些blob块直接引用别的param就可以了。
   */
  void ShareWeights();

  /**
   @brief For an already initialized net, implicitly copies (i.e., using no
           additional memory) the pre-trained layers from another Net.

    当前net对给定的net中具有相同名字的layer进行权值的共享。
   */
  void ShareTrainedLayersWith(const Net* other);

  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string& trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string& trained_filename);
  void CopyTrainedLayersFromHDF5(const string& trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  /**
    @brief 功能描述：从函数名中就可以看出来，它的作用就是过滤掉一些不不符合要求
    layer.
    @param [in]  param          net的参数,这是最初的没有过滤之前的net参数。
    @param [out] param_filtered 过滤之后的net参数。

    具体来说，每一个net中包含了一个net_state的参数,里面指定了该net的phase, level 
    和stage, 每一个layer参数中包含了一个net_state_rule, 里面指定了phase/level/stage
    的值或范围，如果net中的值满足net_state_rule中的值，则该layer就会包含到net中。其
    实这不部分的工作是在 StateMeetsRule函数内完成的。
    */
  static void FilterNet(const NetParameter& param, NetParameter* param_filtered);

  /**
    @brief 根据net_state和net_state_rule来决定一个layer是否符合要求。该函数会在
    FilterNet函数中调用。
    @param [in] state 该参数对应net中的net_state,里面包含了phase/level/stage的值。
    @param [in] rule  该参数对应了layer中的net_state_rule的值,里面包含了pahse/level/
                      stage的的值或范围。
    @param [in] layer_name 一个layer的名字，该参数会在打印日志的时候使用。
    */
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  const vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) {
    before_forward_.push_back(value);
  }
  const vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) {
    after_forward_.push_back(value);
  }
  const vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) {
    after_backward_.push_back(value);
  }

 protected:
  /**
    @brief 该函数实现向net中添加一个top的blob块. 在该函数中会申请要添加的blob块.

    @param [in] param     net的参数，通过它可以获取layer以及top的blob块的相关信息。
    @param [in] layer_id  要添加的layer在ID, 即net中的第几个layer.
    @param [in] top_id    要添加的layer中的top块的ID, 即当前layer中的第几个top块。
    @param [out] available_blobs  这是一个set类型，把新添加的blob块放到里面去。
    @param [out] blob_name_to_idx blob块的名字到它的blobs_列表中的索引值。
    */
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);

  /**
    @brief 该函数实现向net中添加一个bottom的blob块. 在该函数中不需要新申请要添加的bottom
    的blob块, 因为这个blob块在前面的layer执行appendTop时已经申请内存了.

    @param [in] param     net的参数，通过它可以获取layer以及bottom的blob块的相关信息。
    @param [in] layer_id  要添加的layer在ID, 即net中的第几个layer.
    @param [in] bottom_id 要添加的layer中的bottom块的ID, 即当前layer中的第几个bottom块。
    @param [in] available_blobs  这是一个set类型，里面存放了前面的layer中的top块并且这
    些还没有被其它layer层作为bottom的blob块, 也就是这些blob块还没有被其它layer连接起来呢。
    @pram [in] blob_name_to_idx blob块的名字到它的blobs_列表中的索引值。

    @return  返回添加到net中的bottom块的ID索引值(在blobs_中的第几个)
    */
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);

  /**
    @brief 增加一个给定layer的第param_id个的param的blob块到net中，它会被保存到成员变量 params_当中。
    @param [in] param 当前net的参数(netparameter)
    @param [in] layer_id 要增加的是第几个layer.
    @param [in] param_id 要增加是layer中的第几个param块。
    */
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /** @brief 该函数负责打印关于给定layer的前向传播之后的top块的信息，param块的信息等。 */
  void ForwardDebugInfo(const int layer_id);

  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  string name_;         //< 网络的名字
  Phase phase_;         //< TRAIN or TEST
  vector<shared_ptr<Layer<Dtype> > > layers_;   //< 当前net包含所有layers的指针
  vector<string> layer_names_;                  //< 当前net包含所有layers的名字
  map<string, int> layer_names_index_;          
  vector<bool> layer_need_backward_;            //< 每一层的layer是否需要进行反向传播
  
  // 整个net的数据的blob的相关信息，所有的数据的blob块都保存在blobs_中，每一个blob块的对应的相关属性保存在了
  // blob_names_/blob_need_backward_/blob_names_index_等。
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;     // 每一个blob块是否需要反向传播。
  vector<Dtype> blob_loss_weights_;     //求目标函数时每一个blob块的加权值。

  // 整个net中每一层layer的每一个bottom的blob块的指针保存了bottom_vecs_中，每一个blob块对应的相关属性信息保存
  // 在了bootom_id_vecs_/bottom_need_backward_等。
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;      // 每一层layer的每一个bottom块是否需要进行反向传播。
  
  // 整个net中每一层layer的每一个top的blob块的指针保存了top_vecs_中，每一个blob块对应的blob在blobs_中
  // 的索引值保存在了top_id_vecs_中了。
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_; 
  
  // net的输入的blob块的指针以及它们在整个blobs_中的下标索引值，通过这个索引值可以获取
  // 对应blob块的一些属性信息。
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<int> net_input_blob_indices_; 

  // net的输出的blob块指针，以及它们在整个blobs_中的下标索引值，通过这个索引值可以获取
  // 对应blob块的一些属性信息。
  vector<Blob<Dtype>*> net_output_blobs_;
  vector<int> net_output_blob_indices_;
  
  // 下面这几个变量的size()是相同的, net中所有的params_
  vector<shared_ptr<Blob<Dtype> > > params_;        // 网络中全部layer的params(无论共享还是不共享的，者包括着)
  vector<string> param_display_names_;              // 整个net中所有的layer中的param_的名字集合
  vector<int> param_owners_;                        // 每一个param块对应的真正拥有者的ID(在params_中的索引值)
  vector<int> learnable_param_ids_;                 // 一个pram块对应的learnable_parm(因为权值共享的原因)在learnable_params_中的索引
  vector<pair<int, int> > param_layer_indices_;     // layer_id 和 param_id
  vector<vector<int> > param_id_vecs_;              // 每一layer中每一个param中id.(该id就是在params_中的下标值)
  map<string, int> param_names_index_;              // 只会存放有名字的param块，用于了权值共享时，查找一个param是否已经存在

  //  下面几个变量的size()是相同的， 等于网络中非共享的param的个数, 这些参数是可以更新的。
  vector<Blob<Dtype>*> learnable_params_;
  vector<bool> has_params_lr_;
  vector<float> params_lr_;
  vector<bool> has_params_decay_;
  vector<float> params_weight_decay_;

  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  // Callbacks
  vector<Callback*> before_forward_;        // 在正向传播之前要执行的动作
  vector<Callback*> after_forward_;         // 在正向传播之后要执行的动作
  vector<Callback*> before_backward_;       // 在反向传播之前要执行的动作
  vector<Callback*> after_backward_;        // 在反向传播之后要执行的动作

DISABLE_COPY_AND_ASSIGN(Net);
};

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
