#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include "../../../../../../../../.envs/Cellar/llvm/8.0.0/include/c++/v1/vector"

#include "../../blob.hpp"
#include "../../data_transformer.hpp"
#include "../../internal_thread.hpp"
#include "../../layer.hpp"
#include "base_data_layer.hpp"
#include "../../../../builds/release/include/caffe/proto/caffe.pb.h"
#include "../../util/db.hpp"

namespace caffe {

  template<typename Dtype>
  class DataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
    explicit DataLayer(const LayerParameter &param);

    virtual ~DataLayer();

    virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "Data"; }

    virtual inline int ExactNumBottomBlobs() const { return 0; }

    virtual inline int MinTopBlobs() const { return 1; }

    virtual inline int MaxTopBlobs() const { return 2; }

  protected:
    void Next();

    bool Skip();

    virtual void load_batch(Batch<Dtype> *batch);

    shared_ptr<db::DB> db_;
    shared_ptr<db::Cursor> cursor_;
    uint64_t offset_;
  };

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
