//
// Created by Troy Liu on 2019-05-14.
//

#include "caffe/layers/data/data_plus_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/data.hpp"
#include <lz4.h>

namespace caffe {

  template<typename Dtype>
  void DataPlusLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    const int batch_size = this->layer_param().data_param().batch_size();
    DatumPlus datum;
    datum.ParseFromString(this->cursor_->value());
    if (datum.compressed()) {
      string decompressed_data = decompress(datum.data());
      datum.clear_data();
      datum.set_data(decompressed_data);
    }
    Datum shape_datum;
    convert_datum(datum, shape_datum);

    // data
    vector<int> top_shape = this->data_transformer_->InferBlobShape(shape_datum);
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);
    this->transformed_data_.Reshape(top_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(top_shape);
    }
    LOG_IF(INFO, Caffe::root_solver())
                    << "output data size: " << top[0]->num() << ","
                    << top[0]->channels() << "," << top[0]->height() << ","
                    << top[0]->width();

    // label
    if (this->output_labels_) {
      vector<int> label_shape;
      label_shape.push_back(batch_size);
      label_shape.push_back(datum.label_size());
      top[1]->Reshape(label_shape);
      for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
      }
    }
  }

  template<typename Dtype>
  void DataPlusLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    const int batch_size = this->layer_param_.data_param().batch_size();

    DatumPlus datum;
    Datum data;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      timer.Start();
      while (this->Skip()) {
        this->Next();
      }
      datum.ParseFromString(this->cursor_->value());
      if (datum.compressed()) {
        string decompressed_data = decompress(datum.data());
        datum.clear_data();
        datum.set_data(decompressed_data);
      }
      convert_datum(datum, data);
      read_time += timer.MicroSeconds();

      if (item_id == 0) {
        // Reshape according to the first datum of each batch
        // on single input batches allows for inputs of varying dimension.
        // Use data_transformer to infer the expected blob shape from datum.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(data);
        this->transformed_data_.Reshape(top_shape);
        // Reshape batch according to the batch_size.
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape);
      }

      // Apply data transformations (mirror, scale, crop...)
      timer.Start();
      int offset = batch->data_.offset(item_id);
      Dtype *top_data = batch->data_.mutable_cpu_data();
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(data, &(this->transformed_data_));

      // Copy label.
      if (this->output_labels_) {
        Dtype *top_label = batch->label_.mutable_cpu_data();
        for (int j = 0; j < datum.label_size(); ++j) {
          top_label[item_id] = datum.label(j);
        }
      }
      trans_time += timer.MicroSeconds();
      this->Next();
    }
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  }

  INSTANTIATE_CLASS(DataPlusLayer);

  REGISTER_LAYER_CLASS(DataPlus);
}

