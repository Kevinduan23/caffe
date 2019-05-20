//
// Created by Troy Liu on 2019-05-19.
//

#include <string>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/data.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;
using boost::shared_ptr;

template<typename TypeParam>
class DatumPlusTest : public ::testing::Test {
protected:
  DatumPlusTest()
      : root_images_(string(EXAMPLES_SOURCE_DIR) + string("images/cat.jpg")) {}

  virtual void SetUp() {
    img = cv::imread(root_images_);
  }

  virtual ~DatumPlusTest() {}
  string root_images_;
  cv::Mat img;
};

TYPED_TEST_CASE(DatumPlusTest, TestDtypes);

TYPED_TEST(DatumPlusTest, TestCompress) {
  DatumPlus datum;
  const cv::Mat &im = this->img;
  caffe::CVMatToDatum(im, &datum, true);
  DECOMPRESS_AND_SET(&datum)
  cv::Mat decompressed(im.rows, im.cols,
                       im.type(),
                       const_cast<char *>(datum.data().data()));
  EXPECT_EQ(sum(im != decompressed), cv::Scalar(0, 0, 0, 0));
}
}  // namespace caffe