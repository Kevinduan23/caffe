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
#include "caffe/util/rng.hpp"

using caffe::DatumPlus;
using caffe::compress;
using caffe::decompress;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;
namespace bfs = boost::filesystem;

void create_dataset(const string &root_folder,
                    const string &output_folder,
                    const string &db_type,
                    const string &phase) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/nsfw_" + phase + "_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  bfs::directory_iterator root(root_folder);
  float label = 0;
  int fileid = 0;
  int resized_height = 224;
  int resized_width = 224;
  std::vector<std::pair<string, float>> imgs_labels;
  string data;
  DatumPlus datum;
  // get all files
  for (const bfs::directory_entry &dir: root) {
    if (bfs::is_directory(dir)) {
      LOG(INFO) << "Entering "
                << dir.path().string();
      for (const bfs::directory_entry &file: bfs::directory_iterator(dir.path())) {
        auto ext = boost::algorithm::to_lower_copy(bfs::extension(file.path()));
        if (bfs::is_regular_file(file.status())
            && (ext == ".jpg" || ext == ".png")) {
          imgs_labels.emplace_back(std::make_pair(file.path().string(), label));
        }
      }
      label++;
    }
  }
  LOG(INFO) << "Get " << imgs_labels.size() << "images in total";

  // shuffle
  caffe::shuffle(imgs_labels.begin(), imgs_labels.end());

  for (auto &p : imgs_labels) {
    auto path = p.first;
    auto l = p.second;
    datum.clear_height();
    datum.clear_width();
    datum.clear_channels();
    datum.clear_data();
    datum.clear_label();

    datum.set_height(resized_height);
    datum.set_width(resized_width);
    datum.set_channels(3);

    bool success = caffe::ReadImageToDatum(path,
                                           {l},
                                           resized_height,
                                           resized_width,
                                           true,
                                           string("jpg"),
                                           &datum,
                                           false);
    if (!success)
      continue;
    string value;
    CHECK(datum.SerializeToString(&value));

    txn->Put(caffe::format_int(fileid, 8), value);
    ++fileid;
    if (fileid % 5000 == 0) {
      txn->Commit();
    }
  }

  txn->Commit();
  train_db->Close();
}

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  if (argc != 5) {
    std::cout << "Usage:\n"
                 "    create_nsfw_dataset root_folder output_folder db_type\n"
                 "Where the input folder should contain the binary batch files.\n"
                 "You should gunzip them after downloading.\n" << std::endl;
  } else {
    google::InitGoogleLogging(argv[0]);
    create_dataset(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]));
  }
  return 0;
}
