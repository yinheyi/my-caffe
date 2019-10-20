#ifndef CAFFE_UTIL_FORMAT_H_
#define CAFFE_UTIL_FORMAT_H_

#include <iomanip>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>

namespace caffe {

 /**
   @brief 功能描述：该函数实现把数int类型的数字转换为字符串。 并且不满足字符宽度的使用0来填充。
   @param [in] n                      要转换的整数
   @param [in] numberOfLeadingZeros   表示的显示宽度
   @return 返回值是std::string类型, 即格式化之后的字符串。
   */
inline std::string format_int(int n, int numberOfLeadingZeros = 0 ) {
  std::ostringstream s;
  s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
  return s.str();
}

}

#endif   // CAFFE_UTIL_FORMAT_H_
