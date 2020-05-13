#ifndef LBP_MATCHBOX_UTILS_HPP
#define LBP_MATCHBOX_UTILS_HPP

#include <string>
#include <fstream>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include "fileutil.hpp"

using namespace mxnet::cpp;

bool check_datafiles(const std::vector<std::string> &data_files);

bool
setDataIter(MXDataIter *iter, const std::string &useType, const std::vector<std::string> &data_files, int batch_size);

/**
   * Load checkpoint
   *
   * @param filepath
   * @param exe
   */
void LoadCheckpoint(const std::string &param_path, Executor *exe);

/**
 * Save current check point, model.py
 */
void SaveCheckpoint(const std::string &param_path, const std::string &model_path, Symbol &symbol, Executor *exe);

#endif //LBP_MATCHBOX_UTILS_HPP
