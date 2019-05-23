//
// Created by Troy Liu on 2019-05-14.
//

#ifndef CAFFE_DATA_HPP
#define CAFFE_DATA_HPP

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include "caffe/proto/caffe.pb.h"

#define SET_LABELS(datumPlus, labels) std::for_each((labels).cbegin(), (labels).cend(), [datumPlus](const float l) { (datumPlus)->add_label(l); });

#define COMPRESS_AND_SET(datum, data, should_compress)  { \
  if (should_compress && !(datum)->compressed()) (datum)->set_data(caffe::compress(data)); \
  else {(datum)->set_data(data);} \
  (datum)->set_compressed(should_compress); \
}

#define DECOMPRESS_AND_SET(datum) {if ((datum)->compressed()) { \
  (datum)->set_data(caffe::decompress((datum)->data())); \
  (datum)->set_compressed(false);}}

namespace caffe {
void convert_datum(const DatumPlus &from, Datum &to);

std::string compress(const std::string &data);

std::string decompress(const std::string &data);

}

#endif //CAFFE_DATA_HPP
