#include <fstream>
#include "fileutil.hpp"

fs::path getDataDirectory(std::initializer_list<std::string_view> folders)
{
    auto path = fs::current_path();
    path /= "data";
    return std::accumulate(folders.begin(), folders.end(), path,
                               [](fs::path val, std::string_view s) {
                                   return val /= s.data();
                               });
}

bool FileExists(const std::string &filename) {
    std::ifstream fhandle(filename.c_str());
    return fhandle.good();
}
