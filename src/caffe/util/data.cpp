//
// Created by Troy Liu on 2019-05-14.
//

#include "caffe/util/data.hpp"
#include <sstream>

namespace caffe {
void convert_datum(const DatumPlus &from, Datum &to) {
  to.set_channels(from.channels());
  to.set_width(from.width());
  to.set_height(from.height());
  to.set_data(from.data());
  to.set_encoded(from.encoded());
  to.set_label(static_cast<int>(from.label(0)));
}

std::string compress(const std::string &data) {
  namespace bio = boost::iostreams;
  std::stringstream compressed;
  std::stringstream origin(data);
  bio::filtering_streambuf<bio::input> out;
  out.push(
      bio::gzip_compressor(bio::gzip_params(bio::gzip::default_compression)));
  out.push(origin);
  bio::copy(out, compressed);
  return compressed.str();
}

std::string decompress(const std::string &data) {
  namespace bio = boost::iostreams;
  std::stringstream compressed(data);
  std::stringstream decompressed;
  bio::filtering_streambuf<bio::input> out;
  out.push(bio::gzip_decompressor());
  out.push(compressed);
  bio::copy(out, decompressed);
  return decompressed.str();
}
} // namespace caffe
