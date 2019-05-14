//
// Created by Troy Liu on 2019-05-14.
//

#ifndef CAFFE_DATA_HPP
#define CAFFE_DATA_HPP

#include <lz4.h>
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  void convert_datum(const DatumPlus &from, Datum &to);
}

#endif //CAFFE_DATA_HPP
