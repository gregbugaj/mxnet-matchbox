#ifndef MATCHBOX_FILEUTIL_HPP
#define MATCHBOX_FILEUTIL_HPP

#include <experimental/filesystem>
#include <string>
#include <iostream>
#include <numeric>

namespace fs = std::experimental::filesystem;
using namespace std::string_literals;

/// Get current data directory
/// Usage :: getDataDirectory({"folder_a", "folder_b"})
/// \param folders
/// \return
fs::path getDataDirectory(std::initializer_list<std::string_view> folders);

#endif //MATCHBOX_FILEUTIL_HPP
