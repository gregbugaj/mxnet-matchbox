#ifndef LBP_MATCHBOX_UTILS_HPP
#define LBP_MATCHBOX_UTILS_HPP

#include <string>
#include <fstream>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>

using namespace mxnet::cpp;

#define TRY \
  try {
#define CATCH \
  } catch(dmlc::Error &err) { \
    LG << "Status: FAIL";\
    LG << "With Error: " << MXGetLastError(); \
    return 1; \
  }

bool isFileExists(const std::string &filename) {
    std::ifstream fhandle(filename.c_str());
    return fhandle.good();
}

bool check_datafiles(const std::vector<std::string> &data_files) {
    for (size_t index = 0; index < data_files.size(); index++) {
        if (!(isFileExists(data_files[index]))) {
            LG << "Error: File does not exist: " << data_files[index];
            return false;
        }
    }
    return true;
}

bool setDataIter(MXDataIter *iter, const std::string &useType,
                 const std::vector<std::string> &data_files, int batch_size) {
    if (!check_datafiles(data_files)) {
        return false;
    }

    iter->SetParam("batch_size", batch_size);
    iter->SetParam("shuffle", 1);
    iter->SetParam("flat", 1);

    if (useType == "Train") {
        iter->SetParam("image", data_files[0]);
        iter->SetParam("label", data_files[1]);
    } else if (useType == "Label") {
        iter->SetParam("image", data_files[2]);
        iter->SetParam("label", data_files[3]);
    } else {
        throw std::runtime_error("Unknown useType");
    }

    iter->CreateDataIter();
    return true;
}

/**
   * Load checkpoint
   *
   * @param filepath
   * @param exe
   */
void LoadCheckpoint(const std::string &filepath, Executor *exe) {

    std::cerr << "Loading the model parameters." << std::endl;
    std::map<std::string, NDArray> params = NDArray::LoadToMap(filepath);

    for (auto &iter : params) {

        auto type = iter.first.substr(0, 4);
        auto name = iter.first.substr(4);
        std::cerr << "Type/name : " << type << ", " << name << std::endl;

        NDArray target;
        if (type == "arg:")
            target = exe->arg_dict()[name];
        else if (type == "aux:")
            target = exe->aux_dict()[name];
        else
            continue;
        iter.second.CopyTo(&target);
    }
}

/**
 * Save current check point
 * @param filepath
 * @param net
 * @param exe
 */
void SaveCheckpoint(const std::string &filepath, Symbol &net, Executor *exe) {
    auto save_args = exe->arg_dict();
    /*we do not want to save the data and label*/
    save_args.erase("data");
    save_args.erase("data_label");

    // copy any aux array
    for (auto iter : exe->aux_dict()) {
        save_args.insert({"aux:" + iter.first, iter.second});
    }
    NDArray::Save(filepath, save_args);
}

#endif //LBP_MATCHBOX_UTILS_HPP
