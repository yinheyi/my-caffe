#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Return true iff the net is not the current version.
bool NetNeedsUpgrade(const NetParameter& net_param);

// Check for deprecations and upgrade the NetParameter as needed.
bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param);

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const NetParameter& v0_net_param, NetParameter* net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the V1LayerParameter format.
bool UpgradeV0LayerParameter(const V1LayerParameter& v0_layer_connection,
                             V1LayerParameter* layer_param);

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(NetParameter* net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

// Return true iff the Net contains input fields.
bool NetNeedsInputUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade input fields into layers.
void UpgradeNetInput(NetParameter* net_param);

// Return true iff the Net contains batch norm layers with manual local LRs.
bool NetNeedsBatchNormUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade batch norm layers.
void UpgradeNetBatchNorm(NetParameter* net_param);

/** 
  @brief 该函数负责检测solver_param中是否包含了之前老版本中支持的solver_type参数，
  它们是使用枚举类型表示的，新版本都使用字符串表示了。如果包含了老版本的参数，则
  返回true, 意思就是需要进行solver的类型描述upgrade.
  */
bool SolverNeedsTypeUpgrade(const SolverParameter& solver_param);

/**
  @brief 该函数把solver参数中存在的旧版本中的枚举类型的solver_type更改为新版本中的
  字符串表示的类型。
  */
bool UpgradeSolverType(SolverParameter* solver_param);

/**
  @brief 升级SolverParam中的需要修改的内容。原因是为了保持向后兼容，支持之前已经不使用的
  参数. 目前主要是一条：在之前的版本中，Solver的类型是使用枚举类型定义的，现在修改为了使用
  字符串来表示了。(例如：SGD, Nesterov, AdaGrad, RMSProp等)
  */
bool UpgradeSolverAsNeeded(const string& param_file, SolverParameter* param);

/**
  @brief 从文件中读到solver的参数，并解析到SolverParameter类对象来存储, 它还会做一些额外必要
  的情况，包含为了向后兼容性修改一些旧版本中的内容，以及在不提供snapshot的文件前缀或只提供了
  路径的情况下，应该设置一个默认的值。
  @param [in]  param_file 保存solver参数的文件位置。
  @param [out] param      SolverParam的Proto的Message类指针。
  */
void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                       SolverParameter* param);

}  // namespace caffe

#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
