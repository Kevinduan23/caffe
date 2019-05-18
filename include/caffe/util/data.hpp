//
// Created by Troy Liu on 2019-05-14.
//

#ifndef CAFFE_DATA_HPP
#define CAFFE_DATA_HPP

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <lz4.h>
#include "caffe/proto/caffe.pb.h"

#define SET_LABELS(labels, datumPlus) std::for_each(labels.cbegin(), labels.cend(), [datumPlus](const float l) { datumPlus->add_label(l); });

namespace caffe {
void convert_datum(const DatumPlus &from, Datum &to);

std::string compress(const std::string &data);

std::string decompress(const std::string &data);

}

#endif //CAFFE_DATA_HPP
