//
// Created by Troy Liu on 2019-05-17.
//

#include "caffe/layers/accuracy_generic_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define BATCH_SUM(mse, bs_size, bs_num) {\
  Dtype bs_sum; \
  for (int bs_idx = 0; bs_idx < bs_num; ++bs_idx) { \
    bs_sum = caffe_cpu_asum(bs_size, mse + (bs_idx * bs_size)); \
    mse[bs_idx * bs_size] = bs_sum; \
  } \
}

namespace caffe {
template<typename Dtype>
void AccuracyGenericLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  this->acc_type = this->layer_param().accuracy_generic_param().type();
}

template<typename Dtype>
void AccuracyGenericLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  switch (this->acc_type) {
  case AccuracyGenericParameter_Type_SSIM:
  case AccuracyGenericParameter_Type_PSNR: {
    vector<int> top_shape(0);
    top[0]->Reshape(top_shape);
    break;
  }
  default:
    LOG(FATAL) << "Unknown Accuracy Type";
    break;
  }
}
template<typename Dtype>
void AccuracyGenericLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom.size(), 2);
  switch (this->acc_type) {
  case AccuracyGenericParameter_Type_PSNR: {
    CHECK_EQ(bottom[0]->num_axes(), 4);
    CHECK_EQ(bottom[1]->num_axes(), 4);
    const Dtype *data = bottom[0]->cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    // use label's diff to avoid memory allocation
    Dtype *result = bottom[1]->mutable_cpu_diff();
    // compute MSE
    caffe_sub(bottom[1]->count(), data, label, result);
    caffe_powx(bottom[1]->count(), result, Dtype(2), result);
    const int bs_size = bottom[1]->channels() * bottom[1]->height() * bottom[1]->width();
    const int bs_count = bottom[1]->num();
    BATCH_SUM(result, bs_size, bs_count)
    Dtype max;
    if (!this->layer_param().accuracy_generic_param().has_max()) {
      LOG(INFO) << "max not specified, use 255 instead by default";
      max = Dtype(255);
    } else {
      max = Dtype(this->layer_param().accuracy_generic_param().max());
    }

    // compute PNSR every item in one batch
    for (int i = 0; i < bs_count; ++i) {
      *result += 20 * log10(max / (sqrt(result[i * bs_size])));
    }
    top[0]->mutable_cpu_data()[0] = *result / bs_count;
    break;
  }
  case AccuracyGenericParameter_Type_SSIM: {
    CHECK_EQ(bottom[0]->num_axes(), 4);
    CHECK_EQ(bottom[1]->num_axes(), 4);
    const Dtype *data = bottom[0]->cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    // use label's diff to avoid memory allocation


    break;
  }
  default:
    break;
  }
}

#ifdef CPU_ONLY

STUB_GPU(AccuracyGenericLayer);
#endif

INSTANTIATE_CLASS(AccuracyGenericLayer);

REGISTER_LAYER_CLASS(AccuracyGeneric);

} // namespace caffe