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
  case AccuracyGenericParameter_Type_PSNR:Blob<Dtype> mse;
    mse.CopyFrom(*bottom[0], false, true);
    caffe_gpu_sub(mse.count(), bottom_data, bottom_label, mse.mutable_gpu_data());
    caffe_gpu_powx(mse.count(), mse.gpu_data(), Dtype(2), mse.mutable_gpu_data());
    Dtype sum_mse;
    caffe_gpu_asum(mse.count(), mse.cpu_data(), &sum_mse);
    sum_mse /= (mse.width() * mse.height() * mse.num());
    Dtype max;
    if (!this->layer_param().accuracy_generic_param().has_max()) {
      LOG(WARNING) << "max not specified, use 255 instead by default";
      max = Dtype(255);
    } else {
      max = Dtype(this->layer_param().accuracy_generic_param().max());
    }
    Dtype pnsr = 20 * log(10 * (max / (sqrt(sum_mse))));
    top[0]->mutable_cpu_data()[0] = pnsr / mse.num();
    break;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyGenericLayer);
} // namespace caffe