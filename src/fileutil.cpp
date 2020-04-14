#include "fileutil.hpp"

fs::path getDataDirectory(std::initializer_list<std::string_view> folders)
{
    auto path = fs::current_path();
    path /= "../data";
    return std::accumulate(folders.begin(), folders.end(), path,
                               [](fs::path val, std::string_view s) {
                                   return val /= s.data();
                               });
}