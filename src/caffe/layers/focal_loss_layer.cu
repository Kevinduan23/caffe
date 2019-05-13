//
// Created by troy on 19-5-13.
//

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void
FocalLossForwardGPU(const int nthreads, const Dtype *prob_data,
                    const Dtype *label, Dtype *loss, const int num,
                    const int dim, const int spatial_dim,
                    const bool has_ignore_label_, const int ignore_label_,
                    Dtype *counts, float alpha_, float gamma_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim; // batch index
    const int s = index % spatial_dim; // index of pred
    const int label_value = static_cast<int>(label[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      // loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim +
      // s],
      //                Dtype(FLT_MIN)));
      Dtype pt = prob_data[i * dim + s];
      loss[index] =
          -alpha_ * pow(1.0 - pt, gamma_) * log(max(pt, Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *input_data = bottom[0]->gpu_data();
  const Dtype *target = bottom[1]->gpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype *loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype *counts = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  FocalLossForwardGPU<Dtype>
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, input_data, target, loss_data, outer_num_, dim, inner_num_,
          has_ignore_label_, ignore_label_, counts, alpha_, gamma_);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  Dtype normalizer = this->get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
  if (top.size() == 2) {
    top[1]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
__global__ void
FocalLossBackwardGPU(const int nthreads, const Dtype *top, const Dtype *label,
                     Dtype *bottom_diff, const int num, const int dim,
                     const int spatial_dim, const bool has_ignore_label_,
                     const int ignore_label_, Dtype *counts, float alpha_,
                     float gamma_) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim; // batch index
    const int s = index % spatial_dim; // index of prob
    const int label_value = static_cast<int>(label[i]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[i * dim + c] = 0;
      }
    } else {
      Dtype pt = bottom_diff[i * dim + s];
      if (s == label_value) {
        bottom_diff[i * dim + s] =
            alpha_ * gamma_ * log(max(pt, Dtype(FLT_MIN))) *
                pow(1.0 - pt, gamma_ - 1) -
            (alpha_ * pow(1.0 - pt, gamma_)) / max(pt, Dtype(FLT_MIN));
      } else {
        bottom_diff[i * dim + s] =
            -((alpha_ * pow(pt, gamma_)) / (pt - 1.0)) -
            alpha_ * pow(pt, gamma_ - 1) * gamma_ *
                log(max(Dtype(1.0 - pt), Dtype(FLT_MIN)));
      }
      ++(*counts);
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype *prob_data = bottom[0]->gpu_data();
    const Dtype *top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(bottom[0]->count() * sizeof(Dtype), prob_data,
                     bottom_diff);
    const Dtype *label = bottom[1]->gpu_data();
    const int dim = bottom[0]->count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype valid_counts;
    // NOLINT_NEXT_LINE(whitespace/operators)
    FocalLossBackwardGPU<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_data, label, bottom_diff, outer_num_, dim, inner_num_,
            has_ignore_label_, ignore_label_, &valid_counts, alpha_, gamma_);
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    Dtype normalizer = this->get_normalizer(normalization_, valid_counts);
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
    caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);

}