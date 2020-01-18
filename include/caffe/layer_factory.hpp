/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
  /** @brief  定义了一个Creator的函数指针类型，它的参数是LayerParamter, 返回值是指向
    Layer的智能指针.  */
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  /** @brief 定义了一个map类型，用于存放layer的创建者:  它的key是layer的名字，值是相
    应的layer创建函数的指针。*/
  typedef std::map<string, Creator> CreatorRegistry;

  /** @brief 该函数返回一个new 的registry的对象.

    在该函数内,由于g_registry静态的，所以它只会被new一次。有一点不明白的地方，为什么
    不直接写成类对象的形式，而是new一个指针的形式呢？ 难道是为了节约那么一点点的空间？
    因为如果写成对象的形式，在程序运行之前就会被构造出来，如果写成new的形式，只会在调
    用该函数时才被new出来。
   */
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  /** @brief 功能描述： 该函数实现向creator注册表中添加一个新的项。
      @param [in] type 新注册的函数的类型名字，使用字符串表示。
      @param [in] creator 新注册的函数指针.
   */
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    /* 从这里可以看出来，如果一个type已经被注册过了，它就是给出提示信息。我现在不知道
       CHECK_EQ会不会终止程序的运行，如果不终止程序的运行，则新注册的函数会覆盖原来已
       经注册过后函数。 */
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  /** @brief 功能描述: 该函数使用LaperParamter作为参数，创建一个对应的layer, 并返回
      相应的指针. 
      @param [in] param 该参数里面包含了layer的类型和创建layer所需要的数据。

      具体实现：该函数首先通过layer的type在creator注册表中找到相应的layer creator，
      然后调用该函数创建相应的layer. 
   */
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

  /** @brief 功能描述： 该函数获取creator 注册表中所有type的字符串的值。
      @return 返回是一个存放string类型的vector.
   */
  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  /** @brief 把该类的构造函数设置为private类型，阻止创建任何它的实例，因为完全可以
      通过该类内部的静态成员函数来满足我们的需求。*/
  LayerRegistry() {}

  /** @brief 功能描述： 该函数的功能是把creator注册表中的所有类型名都以字符串的形式打印
      出来。 */
  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};


/** @brief  该类仅仅是对AddCreator函数的一层封装啊，为什么要这么搞呢，看不明白啊。

    后补：通过看下面的代码知道了什么单独弄一个类来封装的原因了，目的是让一个layer
    只注册一次,因为通过宏进行注册时会生成一个对应的类对象，如果多次注册同一个layer
    的话，就会生成多个同名的变量了，这样就会报错了。
  */
template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};


/** @brief 功能描述：该宏的作用是把创建类型为type的函数指针添加到注册表中.
    @param [in] 要注册的layer的类型名。
    @param [in] 创建相应layer的函数指针。
 */
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

/** @brief 功能描述： 该宏首先定义了一个creator的函数，然后把该函数与type注册
    的creator注册表中。 
    @param [in] type 要注释的layer的类型名字。

    从该宏的实现中可以看出来，调用该宏的前提时，应该已经定义好了名了typeLayer的
    类。 
  */
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
