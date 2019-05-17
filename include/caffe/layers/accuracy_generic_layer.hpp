//
// Created by Troy Liu on 2019-05-17.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_ACCURACY_GENERIC_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_ACCURACY_GENERIC_LAYER_HPP

#include "caffe/layer.hpp"

namespace caffe {
template<typename Dtype>
class AccuracyGenericLayer : public Layer<Dtype> {
public:
  explicit AccuracyGenericLayer(const LayerParameter &param) : Layer<Dtype>(param) {}

  void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  void Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  const char *type() const override { return "AccuracyGeneric"; };

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  void Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
    for (const auto &i : propagate_down) {
      if (i)
        NOT_IMPLEMENTED;
    }
  }

  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

  AccuracyGenericParameter_Type acc_type;
};
} // namespace caffe

#endif //CAFFE_INCLUDE_CAFFE_LAYERS_ACCURACY_GENERIC_LAYER_HPP
