//
// Created by Troy Liu on 2019-05-03.
//

#ifndef CAFFE_REGION_LOSS_LAYER_HPP
#define CAFFE_REGION_LOSS_LAYER_HPP

#include <vector>
#include <string>
#include <map>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {
  template<typename Dtype>
  Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);

  template<typename Dtype>
  Dtype calc_iou(const vector<Dtype> &box, const vector<Dtype> &truth);

  template<typename Dtype>
  void disp(Blob<Dtype> &swap);

  template<typename Dtype>
  inline Dtype sigmoid(Dtype x) {
    return 1. / (1 + exp(-x));
  }

  template<typename Dtype>
  Dtype softmax_region(Dtype *input, int classes, int stride);

  template<typename Dtype>
  void get_region_box(vector<Dtype> &b, Dtype *x, vector<Dtype> biases, int n, int index,
                      int i, int j, int w, int h, int stride);

  template<typename Dtype>
  Dtype delta_region_box(vector<Dtype> truth, Dtype *x, vector<Dtype> biases, int n, int index,
                         int i, int j, int w, int h, Dtype *delta, float scale, int stride);

  template<typename Dtype>
  void delta_region_class(Dtype *input_data, Dtype *diff, int index, int class_label, int classes, float scale,
                          Dtype *avg_cat);

  template<typename Dtype>
  struct PredictionResult {
    Dtype x;
    Dtype y;
    Dtype w;
    Dtype h;
    Dtype objScore;
    Dtype classScore;
    Dtype confidence;
    int classType;
  };

  struct AvgRegionScore {
    float avg_anyobj;
    float avg_obj;
    float avg_iou;
    float avg_cat;
    float recall;
    float recall76;
    float loss;
  };

  template<typename Dtype>
  class RegionLossLayer : public LossLayer<Dtype> {
  public:
    explicit RegionLossLayer(const LayerParameter &param) : LossLayer<Dtype>(param), diff_() {}

    const char *type() const override { return "RegionLoss"; };

    void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

    void Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

  protected:
    void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) override;

    void Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                      const vector<Blob<Dtype> *> &bottom) override;

    int side_;
    int bias_match_;
    int num_classes_;
    int coords_;
    int num_;
    int softmax_;
    float jitter_;
    int rescore_;

    float object_scale_;
    float class_scale_;
    float noobject_scale_;
    float coord_scale_;

    int absolute_;
    float thresh_;
    int random_;
    vector<Dtype> biases_;

    Blob<Dtype> diff_;
    Blob<Dtype> real_diff_;
    Blob<Dtype> swap_;
    string class_map_;
    map<int, int> cls_map_;
    AvgRegionScore score_;
  };
}

#endif //CAFFE_REGION_LOSS_LAYER_HPP
