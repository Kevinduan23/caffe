//
// Created by Troy Liu on 2019-05-14.
//

#ifndef CAFFE_DATA_PLUS_LAYER_HPP
#define CAFFE_DATA_PLUS_LAYER_HPP

#include "caffe/layers/data/data_layer.hpp"

namespace caffe {
  template<typename Dtype>
  class DataPlusLayer : public DataLayer<Dtype> {

  public:
    explicit DataPlusLayer(const LayerParameter &param) : DataLayer<Dtype>(param) {}

    const char *type() const override { return "DataPlus"; }

    void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  protected:
    void load_batch(Batch<Dtype> *batch) override;
  };
} // namespace caffe

#endif //CAFFE_DATA_PLUS_LAYER_HPP
