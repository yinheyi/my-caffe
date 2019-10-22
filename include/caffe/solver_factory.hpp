/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

/**
  @breif 定义了一个Solver的注册器类，它本质上就是保存了一个map，里面保存了
  solver的类型名到创建对应solver类的函数指针, 该map也就相当于注册表了。
  该类提供方法可以加表格内添加新的solver类型以及相应的创建函数;也提供了查询
  的函数接口，给定solve的类型名，返回相应的创建函数指针。
  */
template <typename Dtype>
class SolverRegistry {
 public:

  /** 
    @brief  Cretor函数指针类型的定义.  该函数输入是solverParameter,返回值
    为solver类指针。 */
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);

  /** @brief 定义了一个注册表类型, 它是solver类型到solver的创建函数的映射。 */
  typedef std::map<string, Creator> CreatorRegistry;

  /**
    @brief 该函数返回当前的注册表。注意它的实现，使用了一个静态类型，表明了只存在
    一份注册表。
    */
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  /**
    @brief 注册表新成员的添加函数。
    @param [in] type    新solver的类型
    @param [in] creator 新solver的创建函数
    */
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }

  /**
    @brief 注册表的查询函数, 给定solver的参数，从中提取出solver的类型，然后返回
    相应的solver创建函数。
    */
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    return registry[type](param);
  }

  /**
    @brief 返回注册表内所有的solver的类型名, 返回的是一个vector<string>类型。
    */
  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry() {}

  /** 返回注册表中所有的solver类型名，返回的是一个字符串，用于打印显示出来。 */
  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};


/**
  @brief 定义了一个类，该类的唯一任务就是在构造函数中把 solver的type和cretor注册到
  注册表格中。
 */
template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};


/**
  @brief 该宏通过实例化一个类对象的方式对solver类进行注册。这么做的目的应试是可以有效在
  编译期内检测出多次注册的错误，因为你不能定义两个同名的全局变量吧, 重定义。  
  该宏用于用户自己实现的creator的注册。
  */
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

/**
  @brief 该宏不仅定义了一个相应的solver生成的函数，并且负责把相应的solver类型和新定义的
  函数注册到表格中。 
  该宏用于一个solver类的注册,因为在宏本身内部自己实现了cretor函数。
  */
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
