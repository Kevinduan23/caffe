//
// Created by Troy Liu on 2019-05-11.
//

#ifndef CAFFE_FOCAL_LOSS_LAYER_HPP
#define CAFFE_FOCAL_LOSS_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

  template<typename Dtype>
  class FocalLossLayer : public LossLayer<Dtype> {
  public:
    explicit FocalLossLayer(const LayerParameter &param)
            : LossLayer<Dtype>(param),
              has_ignore_label_(false),
              ignore_label_(0),
              normalization_(LossParameter_NormalizationMode_BATCH_SIZE),
              outer_num_(0),
              inner_num_(0),
              alpha_(0),
              gamma_(0) {}

    const char *type() const override { return "FocalLoss"; }

    void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

    void Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  protected:
    void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

    void Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

    void Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                      const vector<Blob<Dtype> *> &bottom) override;

    void Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                      const vector<Blob<Dtype> *> &bottom) override;

    virtual Dtype get_normalizer(LossParameter_NormalizationMode normalization_mode, int valid_count) {
      Dtype normalizer;
      switch (normalization_mode) {
        case LossParameter_NormalizationMode_FULL:
          normalizer = Dtype(outer_num_ * inner_num_);
          break;
        case LossParameter_NormalizationMode_VALID:
          if (valid_count == -1) {
            normalizer = Dtype(outer_num_ * inner_num_);
          } else {
            normalizer = Dtype(valid_count);
          }
          break;
        case LossParameter_NormalizationMode_BATCH_SIZE:
          normalizer = Dtype(outer_num_);
          break;
        case LossParameter_NormalizationMode_NONE:
          normalizer = Dtype(1);
          break;
        default:
          LOG(FATAL) << "Unknown normalization mode: "
                     << LossParameter_NormalizationMode_Name(normalization_mode);
      }
      // Some users will have no labels for some examples in order to 'turn off' a
      // particular loss in a multi-task setup. The max prevents NaNs in that case.
      return std::max(Dtype(1.0), normalizer);
    }

    /// Whether to ignore instances with a certain label.
    bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    int ignore_label_;
    LossParameter_NormalizationMode normalization_;
    Dtype normalizer_;
    int outer_num_, inner_num_;
  private:
    float alpha_;
    float gamma_;
  };
}

#endif //CAFFE_FOCAL_LOSS_LAYER_HPP
