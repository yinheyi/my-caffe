#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <algorithm>
#include <iterator>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"

#include "caffe/common.hpp"

namespace caffe {

// boost::mt19937是一个产生随机数的生成器,它是根据mersenne_twister算法随机产生随机数用的。
typedef boost::mt19937 rng_t;

/**
  @brief  该函数返回一个随机数生成器，它的类型是一个类对象指针，在该类中重载了()操作符，
  所以可以当作函数使用。

  在实现中，调用了caffe:rng_stream()函数，它是在common.hpp文件中定义的，当去查看时你会发现:
  rng_stream()返回的是一个RNG的类， 而该类的具体实现又交给了Generator类, 而该类的具体实现又
  是交给了rng_t(即boost::mt19937)类的完成的。一层层的包装，看的有点傻眼！
  */
inline rng_t* caffe_rng() {
  return static_cast<caffe::rng_t*>(Caffe::rng_stream().generator());
}

/**
  @brief 使用Fisher-Yates算法来完成对指定区间内元素的洗牌。
  @param [in] begin  迭代器的begin.
  @param [in] end    迭代器的end.
  @param [in] gen    随机数产生器,它应该是一个重载了()的类指针或是一个函数指针, 使用它的结果当作随机种子。

  具体来说，fisher-yates算法就是：一张张地从样本集中随机抽取出一个样本，直到抽取完毕。
  在该函数模板中， 它所支持的容器必须是可以随机访问的，在std容器中只有vector和array具
  在这种特性.
  */
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                    RandomGenerator* gen) {
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
      difference_type;
  typedef typename boost::uniform_int<difference_type> dist_type;

  difference_type length = std::distance(begin, end);
  if (length <= 0) return;

  for (difference_type i = length - 1; i > 0; --i) {
    // 初始化了一个在区间[0,i]的服从均匀分布的随机数产生器。
    dist_type dist(0, i);
    std::iter_swap(begin + i, begin + dist(*gen));
  }
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
  shuffle(begin, end, caffe_rng());
}
}  // namespace caffe

#endif  // CAFFE_RNG_HPP_
