//
// Created by Troy Liu on 2019-05-18.
//

#include <fstream>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/data.hpp"
#include "caffe/util/io.hpp"

using caffe::DatumPlus;
using caffe::compress;
using caffe::decompress;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;
namespace bfs = boost::filesystem;

void create_dataset(const string &root_folder, const string &output_folder, const string &db_type) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/nsfw_test_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  bfs::directory_iterator root(root_folder);
  float label = 0;
  int fileid = 0;
  string data;
  DatumPlus datum;
  for (const bfs::directory_entry &dir: root) {
    if (bfs::is_directory(dir)) {
      LOG(INFO) << "Entering "
                << dir.path().string();
      for (const bfs::directory_entry &file: bfs::directory_iterator(dir.path())) {
        auto ext = boost::algorithm::to_lower_copy(bfs::extension(file.path()));
        if (bfs::is_regular_file(file.status())
            && (ext == ".jpg" || ext == ".png")) {
          datum.clear_height();
          datum.clear_width();
          datum.clear_channels();
          datum.clear_data();
          datum.clear_label();

//          bfs::ifstream ifs(file.path(), std::ios::in | std::ios::binary | std::ios::ate);
//          bfs::ifstream::pos_type fileSize = ifs.tellg();
//          ifs.seekg(0, std::ios::beg);
//          std::vector<char> bytes(fileSize);
//          ifs.read(bytes.data(), fileSize);
//          string data(bytes.data(), fileSize);
//
          cv::Mat pic = cv::imread(file.path().string());
          if (pic.empty())
            continue;
          cv::Mat out;
          if (pic.channels() == 1) {
            cv::cvtColor(pic, out, cv::COLOR_GRAY2BGR);
            cv::imwrite(file.path().string(), out);
          } else if (pic.channels() == 4) {
            cv::cvtColor(pic, out, cv::COLOR_BGRA2BGR);
            cv::imwrite(file.path().string(), out);
          }
//          datum.set_height(pic.rows);
//          datum.set_width(pic.cols);
//          datum.set_channels(pic.channels());
//
//          uint32_t size = pic.channels() * pic.rows * pic.cols;
//          data.resize(size);
//          std::memcpy(&data[0], pic.data, size * sizeof(*pic.data));
//
//          datum.set_data(compress(data));
//          datum.add_label(label);

          caffe::ReadImageToDatum(file.path().string(), {label}, string("jpg"), &datum, false);
          string value;
          CHECK(datum.SerializeToString(&value));

          txn->Put(caffe::format_int(fileid, 8), value);
          if (fileid % 500 == 0) {
            LOG(INFO) << "Commit...";
            txn->Commit();
          }
          fileid++;
        }
      }
      label++;
    }
  }
  txn->Commit();
  train_db->Close();
}

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  if (argc != 4) {
    std::cout << "Usage:\n"
                 "    create_nsfw_dataset root_folder output_folder db_type\n"
                 "Where the input folder should contain the binary batch files.\n"
                 "You should gunzip them after downloading.\n" << std::endl;
  } else {
    google::InitGoogleLogging(argv[0]);
    create_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
