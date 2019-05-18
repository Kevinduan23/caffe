//
// Created by Troy Liu on 2019-05-17.
//

#include "caffe/layers/accuracy_generic_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void AccuracyGenericLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  this->acc_type = this->layer_param().accuracy_generic_param().type();
}

template<typename Dtype>
void AccuracyGenericLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  switch (this->acc_type) {
  case AccuracyGenericParameter_Type_PSNR:
    vector<int> top_shape(0);
    top[0]->Reshape(top_shape);
    break;
//  default:
//    LOG(FATAL) << "Unknown Accuracy Type";
//    break;
  }
}
template<typename Dtype>
void AccuracyGenericLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom.size(), 2);
  switch (this->acc_type) {
  case AccuracyGenericParameter_Type_PSNR:
    CHECK_EQ(bottom[0]->num_axes(), 4);
    CHECK_EQ(bottom[1]->num_axes(), 4);
    const Dtype *data = bottom[0]->cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    // use label's diff to avoid memory allocation
    Dtype *mse = bottom[1]->mutable_cpu_diff();
    caffe_sub(bottom[1]->count(), data, label, mse);
    caffe_powx(bottom[1]->count(), mse, Dtype(2), mse);
    Dtype sum_mse =
        caffe_cpu_asum(bottom[1]->count(), mse) / (bottom[1]->width() * bottom[1]->height() * bottom[1]->num());
    Dtype max;
    if (!this->layer_param().accuracy_generic_param().has_max()) {
      LOG(INFO) << "max not specified, use 255 instead by default";
      max = Dtype(255);
    } else {
      max = Dtype(this->layer_param().accuracy_generic_param().max());
    }
    Dtype pnsr = 20 * log10(max / (sqrt(sum_mse)));
    top[0]->mutable_cpu_data()[0] = pnsr / bottom[1]->num();
    break;
  }
}

#ifdef CPU_ONLY

STUB_GPU(AccuracyGenericLayer);
#endif

INSTANTIATE_CLASS(AccuracyGenericLayer);

REGISTER_LAYER_CLASS(AccuracyGeneric);

} // namespace caffe