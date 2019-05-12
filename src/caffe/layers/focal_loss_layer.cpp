//
// Created by Troy Liu on 2019-05-11.
//

#include "caffe/layers/focal_loss_layer.hpp"
#include <cfloat>

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  alpha_ = this->layer_param().focal_loss_param().alpha();
  gamma_ = this->layer_param().focal_loss_param().gamma();
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize()
                         ? LossParameter_NormalizationMode_VALID
                         : LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  // shape top size to 1
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "Inputs must have the same dimension.";
  outer_num_ = bottom[0]->shape(0); // batch size
  inner_num_ = bottom[0]->count(1); // instance size: |output| == |target|
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *input_data = bottom[0]->cpu_data();
  const Dtype *target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  Dtype pt = 0;
  for (int i = 0; i < outer_num_; ++i) {
    const int label_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      continue;
    }
    DCHECK_GE(label_value, 0);
    DCHECK_LT(label_value, bottom[0]->shape(1));
    for (int j = 0; j < inner_num_; ++j) {
      pt = input_data[i * inner_num_ + j];
      pt = (label_value == j) ? pt : 1.0 - pt;
      loss -=
          alpha_ * pow(1.0 - pt, gamma_) * log(std::max(pt, Dtype(FLT_MIN)));
      ++valid_count;
    }
  }
  Dtype normalizer = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
  if (top.size() == 2) {
    top[1]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype *prob_data = bottom[0]->cpu_data();
    const Dtype *labels = bottom[1]->cpu_data();
    const int dim = bottom[0]->count() / outer_num_;
    int valid_count = 0;
    Dtype focal_diff = 0;
    Dtype pt = 0;
    Dtype pc = 0;
    for (int i = 0; i < outer_num_; ++i) {
      const int label_value = static_cast<int>(labels[i]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        caffe_set(dim, Dtype(0.0), &bottom_diff[i * dim]);
        continue;
      }
      for (int j = 0; j < inner_num_; ++j) {
        pt = prob_data[i * dim + j];
        if (j == label_value) {
          focal_diff =
              alpha_ * gamma_ * log(std::max(pt, Dtype(FLT_MIN))) *
                  pow(1.0 - pt, gamma_ - 1) -
              (alpha_ * pow(1.0 - pt, gamma_)) / std::max(pt, Dtype(FLT_MIN));
        } else {
          focal_diff = -((alpha_ * pow(pt, gamma_)) / (pt - 1.0)) -
                       alpha_ * pow(pt, gamma_ - 1) * gamma_ *
                           log(std::max(Dtype(1.0 - pt), Dtype(FLT_MIN)));
        }
        bottom_diff[i * dim + j] = focal_diff;
        valid_count++;
      }
    }
    Dtype normalizer = get_normalizer(normalization_, valid_count);
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY

STUB_GPU(FocalLossLayer);
#endif
INSTANTIATE_CLASS(FocalLossLayer);

REGISTER_LAYER_CLASS(FocalLoss);
} // namespace caffe
