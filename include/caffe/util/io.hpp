#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

// 这里使用了boost库中的filesystem库，它可以创建/处理目录路径相关。
#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

/**
  @define CAFFE_TMP_DIR_RETRIES
  @brief 定义了一个宏，用于在创建临时路径名是的偿试次数。
  @details 该宏存在的意意义是：因为临时路径名是随机创建的，它可能会出现创建的路径名已经存在的情况，
  这时候会创建失败，所以呢，这时使用到一个偿试次数。
  */
#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

/**
  @brief 功能描述：该函数用于创建一个临时的目录名(即路径名).
  @param [out] temp_dirpaname 输出创建好的目录名。
  @return 返回值是bool类型，指明是否创建成功。
  
   具体实现中，都通过调用boost库中的filesystem库来完成的: 在boost::filesystem库中使用一个类来表
  示path, 这样做特别的灵活，具体很高的移植性。因为一个路径名可以使用多种形式表示，可能是char*, 
  可能是w_char*, 可能是string,可能是迭代器表示的一个区间范围等。  “/" 重载运算符的作用是append！！
  1. 创建一个临时的路径名，在此基本上增加了相同的最后的目录名"caffe_test.%%%%-%%%%".  
  2. 调用unique_path函数随机生成替代字符串来代替%%%%-%%%%部分。
  3. 调用create_directory函数来偿试生成目录，并以string的格式返回创建的目录名。
  */
inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

/**
  @brief 功能描述：生成一个在临时目录下的临时文件名， 返回的文件名是包含了路径的！
  @param [out] temp_filename 用于返回生成的临时文件名。
  @return 返回值为空。
  */
inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;      ///< static类型，用于保存临时的路径名，目录只生成一次就OK了。
  static uint64_t next_temp_file = 0;  ///< static类型，用于计数下一个将要生成的文件名，即文件名依次为0，1，2, ...
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  
  /** 使用到的format_int()函数是在util/format.hpp文件中定义的。*/
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

/**
  @brief 功能描述：该函数实现从文本文件中读取定义好的protocol buffer内容至内存中Message类中, 即相当于解析的过程。
  @param [in] filename 参数格式是char*类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  @return 成功返回true,失败返回false.
  */
bool ReadProtoFromTextFile(const char* filename, Message* proto);

  /**
  @brief 功能描述：该函数实现从文本文件中读取定义好的protocol buffer内容至内存中Message类中, 即相当于解析的过程。
  @param [in] filename 参数格式是string&类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  @return 成功返回true,失败返回false.
  */
inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

  /**
  @brief 功能描述：该函数实现从文本文件中读取定义好的protocol buffer内容至内存中Message类中, 即相当于解析的过程。 如果读取
  失败，则程序终止。
  @param [in] filename 参数格式是char*类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  @return 返回值为空。
  */
inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

  /**
  @brief 功能描述：该函数实现从文本文件中读取定义好的protocol buffer内容至内存中Message类中, 即相当于解析的过程。 如果读取
  失败，则程序终止。
  @param [in] filename 参数格式是string&类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  @return 返回值为空。
  */
inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

 /**
   @brief 把Message类内的数据序列化到文本文件(名字格式通常是*.prototxt)中。
   @param [in] proto     Message&类型，表示要序列化的Message类;
   @param [in] filename  char*类型，表示要保存到的文件路径;
   @return 返回值为空。
   */
void WriteProtoToTextFile(const Message& proto, const char* filename);
  
 /**
   @brief 把Message类内的数据序列化到文本文件(名字格式通常是*.prototxt)中,如果指定的文件不
   存在，则会新建一个文件。
   @param [in] proto     Message&类型，表示要序列化的Message类;
   @param [in] filename  string&类型，表示要保存到的文件路径;
   @return 返回值为空。
   */  
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

/**
  @brief 功能描述：该函数实现从二进制文件中读取定义好的protocol buffer内容至内存中Message类中, 即相当于解析的过程。
  @param [in] filename 参数格式是char*类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  @return 成功返回true,失败返回false.
  */
bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

/**
  @brief 功能描述：该函数实现从二进制文件中读取定义好的protocol buffer内容至内存中Message类中, 即相当于解析的过程。
  @param [in] filename 参数格式是string&类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  @return 成功返回true,失败返回false.
  */
inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}
  
/**
  @brief 功能描述：该函数实现从二进制文件中读取定义好的protocol buffer内容至内存中Message类中, 如果失败，则软件中止。
  @param [in] filename 参数格式是char*类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  */
inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

/**
  @brief 功能描述：该函数实现从二进制文件中读取定义好的protocol buffer内容至内存中Message类中, 如果失败，则软件中止。
  @param [in] filename 参数格式是string类型，用于表示文件的路径名
  @param [in] proto    参数格式是Message类指针， Message类应该是【根据protocol文件生成的所有Message类】的基类(我猜测的)
  */
inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

/**
  @brief 把Message类内的数据序列化到二进制文件中,如果指定的文件不存在，则会新建一个文件。
  @param [in] proto     Message&类型，表示要序列化的Message类;
  @param [in] filename  char*类型，表示要保存到的文件路径;
  @return 返回值为空。
  */  
void WriteProtoToBinaryFile(const Message& proto, const char* filename);

/**
  @brief 把Message类内的数据序列化到二进制文件中,如果指定的文件不存在，则会新建一个文件。
  @param [in] proto     Message&类型，表示要序列化的Message类;
  @param [in] filename  string&类型，表示要保存到的文件路径;
  @return 返回值为空。
  */  
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

 /** 
   @brief 功能描述：从文件中读取数据至Datum结构中，其中Datum是在src/caffe/proto/caffe.proto定义的。
   @param [in]  filename 要读取的文件路径，从中读取data设置到Datum中;
   @param [in]  label    要设置到Datum中的label值;
   @param [out] datum    要设置的Datum指针;
   @return 文件读取成功时就返回true, 失败返回false.
   */
bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

 /** 
   @brief 功能描述：从文件中读取数据至Datum结构中，其中Datum是在src/caffe/proto/caffe.proto定义的. Datum的label
   值默认设置为-1.
   @param [in]  filename 要读取的文件路径，从中读取data设置到Datum中;
   @param [out] datum    要设置的Datum指针;
   @return 文件读取成功时就返回true, 失败返回false.
   */
inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
#endif  // USE_OPENCV

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
