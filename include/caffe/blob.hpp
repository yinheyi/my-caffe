#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;        // blob数据块最大的维度

namespace caffe {

/**
  @brief
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  /** @brief 功能描述： 初始化一个数据块， 该函数已经是废弃，不应该再使用.
      @param [in] num       blob块内的数据维度
      @param [in] channels  blob块内数据维度
      @param [in] height    blob块内数据维度
      @param [in] width     blob块内数据维度 
   */
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  /** @brief 功能描述： 初始化一个数据块.  */
  explicit Blob(const vector<int>& shape);

  /** @brief 功能描述： reshape一个[num * channels * height * width] 的blob块。
      @param [in] num      第一维度的大小
      @param [in] channels 第二维度的大小
      @param [in] height   第三维度的大小
      @param [in] width    第四维度的大小

      额外说明：该函数已经弃用了，不应该 再被使用。 
   */
  void Reshape(const int num, const int channels, const int height, const int width);
  /** @brief 功能描述： reshape一个指定维度的blob块。
      @param [in] shape 该参数是一个vector, 这里存放了维度值,如：[2, 5, 6, 22]

      当reshape时，如果内存空间不足，则会重新申请一块足够的内存，如果blob块之前的内存
      空间(capacity)大时，也不会释放多余的空间。对于blob块，有两个参数：count_和capacity
      count_表示当前的blob块内数据的个数，capacity表示blob块的内存空间大小,  类似于vector
      的size()和capacity().  
   */
  void Reshape(const vector<int>& shape);
  /** @brief 功能描述：通过BlobShape结构传递shape值进行reshape */
  void Reshape(const BlobShape& shape);
  /** @brief 功能描述：通过一个其它的Blob的形状传递shape值进行reshape */
  void ReshapeLike(const Blob& other);
  /** @brief 功能描述： 返回当前blob块的shape的字符串描述。例如：如果当前的shape为[2,3,4],
    则会返回这样的字符串：2 3 4 (12).
    */
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  /** @brief 功能描述： 返回当前blob块的形状大小, 返回值是一个vector。*/
  inline const vector<int>& shape() const { return shape_; }
  /** @brief 该函数返回给定下标的shape的值, 该值可以为负整数，表示倒数第几个。
      @param [in] index 给定的下标值，如果当前的shape维度为7,则index应该是区间[-7, 7)的正
      整数。

      注意：1. 下标是从0表示的. 
            2. 下标为负整数时表示倒数第几个，例如-2,表示倒数第2个。
    */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  /** @brief 功能描述： 返回当前blob块的维度大小。*/
  inline int num_axes() const { return shape_.size(); }
  /** @biref 功能描述： 返回当前blob块的大小。*/
  inline int count() const { return count_; }
  /**
    @brief 功能描述： 该函数返回给定维度区间[start_axis, end_axis)上的大小（体积).
    @param [in] start_axis 开始的维度
    @param [in] end_axis 终止的维度
    */
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
    @brief 功能描述： 该函数返回给定维度区间[start_axis, 末尾)上的大小（体积).
    @param [in] start_axis 开始的维度
    */
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
    @brief 功能描述： 该函数负责把给定在区间[-num_axes, num_axes)上的整数规范化
           到区间[0, num_axes)上。
    @param [in] axis_index 给定的维度索引。

    具体来说，对于在区间[0, num_axes)上的整数保持不变，对区间[-num_axes, 0)上的整
    数值加上num_axes. 例如：如果num_axes为7, 给定了一个-1, 则返回6. 因为倒数第1个
    数就是正数第6个(从0开始).
    */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /** 下面几个函数就是之前代码遗留下来的，，返回num/chanels/height/width的维度大小
      其中LegacyShape()函数负责检测给定的维度索引值是不是有效。*/
  inline int num() const { return LegacyShape(0); }
  inline int channels() const { return LegacyShape(1); }
  inline int height() const { return LegacyShape(2); }
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes())    // 这个地方返回1是合理的。
      return 1;
    return shape(index);
  }

  /** @brief 计算给定坐标数据的偏移量， 这个偏移量用于从内存中取数据啊。
      @param [in] n, c, h, w 四个参数指针了每一维度的坐标值。
    */
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  /** @brief 计算给定坐标数据的偏移量， 这个偏移量用于从内存中取数据啊。
      @param [in] indices 该参数是一个vector, 里面的数据指定了每一维度的坐标值。

      说明两点：
      1.从下面的代码中可以看出来，在blob块中的shape低索引的维度的值变化慢，高索
      引的维度的值变化快，意思就是说数据总是先填满高索引的维度空间。
      2. 对于参数vector, 如果blob为7维的，vector的值为[1,2]时，则默认返回的坐标
      是[1, 2, 0, 0, 0, 0, 0]的偏移值。
    */
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  /** @brief 功能描述：从另一个blob块中拷贝数据至当前blob块。
      @param [in] source    源blob块
      @param [in] copy_diff 当为true时，拷贝diff数据，当为false时，拷贝data数据
      @param [in] reshape   当为true时，进行reshape,否则不进行reshape(此时如果不满足要求
                            会报错。

     疑问：为什么不同时拷贝data和diff呢？
    */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  /** @brief 功能描述： 返回给定坐标处的数据值 */
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }
  /** @brief 功能描述： 返回给定坐标处的数据值 */
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  /** @brief 功能描述： 返回给定坐标处的diff值 */
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }
  /** @brief 功能描述： 返回给定坐标处的diff值 */
  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  /** @brief 功能描述： 返回该blob块中管理data的syncedmemory指针 */
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  /** @brief 功能描述： 返回该blob块中管理diff的syncedmemory指针 */
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  Dtype* mutable_cpu_data();
  void set_cpu_data(Dtype* data);

  const Dtype* gpu_data() const;
  Dtype* mutable_gpu_data();
  void set_gpu_data(Dtype* data);

  const Dtype* cpu_diff() const;
  Dtype* mutable_cpu_diff();

  const Dtype* gpu_diff() const;
  Dtype* mutable_gpu_diff();

  const int* gpu_shape() const;

  /** @brief 该函数更新data的值： new_data = -1 * diff + data. */
  void Update();
  /** @brief 从BlobProto中读取数据进行构造blob块,用于从保存的模型数据中恢复网络模型 */
  void FromProto(const BlobProto& proto, bool reshape = true);
  /** @brief 把blob块的shape和data写到BlobProto中, write_diff控制是否写入diff */
  void ToProto(BlobProto* proto, bool write_diff = false) const;
  /** @brief 计算data的绝对值的和, absolute sum */
  Dtype asum_data() const;
  /** @brief 计算diff的绝对值的和, absolute sum */
  Dtype asum_diff() const;
  /** @brief 计算data的平和的和, square sum */
  Dtype sumsq_data() const;
  /** @brief 计算diff的平和的和, square sum */
  Dtype sumsq_diff() const;
  /** @brief 为data的值乘上一个因子facor */
  void scale_data(Dtype scale_factor);
  /** @brief 为diff的值乘上一个因子facor */
  void scale_diff(Dtype scale_factor);
  /** @brief 共用另一个blob块的data值 */
  void ShareData(const Blob& other);
  /** @brief 共用另一个blob块的diff值。 */
  void ShareDiff(const Blob& other);
  /** @brief 判断两个blob块的shape是否相同。 */
  bool ShapeEquals(const BlobProto& other);

 protected:
  shared_ptr<SyncedMemory> data_;        // blob块中data数据的指针
  shared_ptr<SyncedMemory> diff_;        // blob块中diff数据的指针
  shared_ptr<SyncedMemory> shape_data_;  // blob块中diff数据的指针
  vector<int> shape_;      // 该blob块的形状, 如[2,4,4]
  int count_;              // 该blob块中数据总数
  int capacity_;           // 该blob块的总容量

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
