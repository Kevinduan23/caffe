//
// Created by Troy Liu on 2019-05-14.
//

#include "caffe/util/data.hpp"

namespace caffe {
  void convert_datum(const DatumPlus &from, Datum &to) {
    to.set_channels(from.channels());
    to.set_width(from.width());
    to.set_height(from.height());
    to.set_data(from.data());
    to.set_encoded(from.encoded());
    to.set_label(static_cast<int>(from.label(0)));
  }
}
