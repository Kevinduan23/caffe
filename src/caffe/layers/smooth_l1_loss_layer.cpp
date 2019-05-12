//
// Created by Troy Liu on 2019-05-12.
//

#include "caffe/layers/smooth_l1_loss_layer.hpp"

namespace caffe {
  template<typename Dtype>
  void SmoothL1LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    has_weght_ = (bottom.size() == 3);
  }

  template<typename Dtype>
  void SmoothL1LossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    if (has_weght_) {
      CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
      CHECK_EQ(bottom[0]->height(), bottom[2]->height());
      CHECK_EQ(bottom[0]->width(), bottom[2]->width());
    }
    diff_.Reshape(bottom[0]->shape());
    errors_.Reshape(bottom[0]->shape());
  }

  template<typename Dtype>
  void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    int count = bottom[0]->count();
    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());
    if (has_weght_) {
      caffe_mul(count, bottom[2]->cpu_data(), diff_.cpu_data(), diff_.mutable_cpu_data());
    }

    const Dtype *diff_data = diff_.cpu_data();
    Dtype *error_data = errors_.mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
      Dtype val = diff_data[i];
      Dtype abs_val = fabs(val);
      if (abs_val < 1.0) error_data[i] = 0.5 * pow(abs_val, 2);
      else error_data[i] = abs_val - 0.5;
    }
    top[0]->mutable_cpu_data()[0] = caffe_cpu_asum(count, errors_.cpu_data()) / bottom[0]->shape(0);
  }

  template<typename Dtype>
  void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                              const vector<Blob<Dtype> *> &bottom) {
    int count = diff_.count();
    Dtype *diff_data = diff_.mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
      Dtype val = diff_data[i];
      if (fabs(val) < 1.0) continue;
      else diff_data[i] = val < -1.0 ? -1.0 : 1.0;
    }
    for (int j = 0; j < 2; ++j) {
      if (propagate_down[j]) {
        const Dtype sign = j == 0 ? 1.0 : -1.0;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[j]->shape(0);
        caffe_cpu_axpby(bottom[j]->count(),
                        alpha,
                        diff_.cpu_data(),
                        Dtype(0),
                        bottom[j]->mutable_cpu_diff());
      }
    }
  }

#ifdef CPU_ONLY

  STUB_GPU(SmoothL1LossLayer)

#endif
  INSTANTIATE_CLASS(SmoothL1LossLayer);

  REGISTER_LAYER_CLASS(SmoothL1Loss);
} // namespace caffe