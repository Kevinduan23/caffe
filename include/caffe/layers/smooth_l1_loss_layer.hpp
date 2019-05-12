//
// Created by Troy Liu on 2019-05-12.
//

#ifndef CAFFE_SMOOTH_L1_LOSS_LAYER_HPP
#define CAFFE_SMOOTH_L1_LOSS_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype> class SmoothL1LossLayer : public LossLayer<Dtype> {
public:
  explicit SmoothL1LossLayer(const LayerParameter &param)
      : LossLayer<Dtype>(param) {}

  const char *type() const override { return "SmoothL1Loss"; }

  void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                  const vector<Blob<Dtype> *> &top) override;

  void Reshape(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top) override;

  int MinBottomBlobs() const override { return 2; }

  int MaxBottomBlobs() const override { return 3; }

  bool AllowForceBackward(const int bottom_index) const override {
    return true;
  }

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;

  void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;

  void Backward_cpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;

  void Backward_gpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  bool has_weght_;
};
} // namespace caffe

#endif // CAFFE_SMOOTH_L1_LOSS_LAYER_HPP
