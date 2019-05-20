//
// Created by troy on 19-5-17.
//

#include "caffe/layers/accuracy_generic_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void AccuracyPNSRGPU(const int nthreads,
                                const Dtype *bottom_data, const Dtype *label, Dtype *pnsr) {}

template<typename Dtype>
void AccuracyGenericLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *bottom_label = bottom[1]->gpu_data();
  switch (this->acc_type) {
  case AccuracyGenericParameter_Type_PSNR:
    const Dtype *temp = bottom[1]->mutable_gpu_diff();
    Dtype acc;
    caffe_gpu_sub(bottom[1]->count(), bottom_data, bottom_label, temp);
    caffe_gpu_dot(bottom[1]->count(), temp, temp, &acc)
    acc /= bottom[1]->count();

//    Dtype max;
//    if (!this->layer_param().accuracy_generic_param().has_max()) {
//      LOG(INFO) << "max not specified, use 255 instead by default";
//      max = Dtype(255);
//    } else {
//      max = Dtype(this->layer_param().accuracy_generic_param().max());
//    }
    acc = 10 * log10(Dtype(1) / acc);
    top[0]->mutable_cpu_data()[0] = acc;
    break;
  }
}

template<typename Dtype>
void AccuracyGenericLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                               const vector<bool> &propagate_down,
                                               const vector<Blob<Dtype> *> &bottom) override {
  for (const auto &i: propagate_down) {
    if (i)
      NOT_IMPLEMENTED;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyGenericLayer);
} // namespace caffe