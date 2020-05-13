#ifndef MATCHBOX_VECTORUTILS_HPP
#define MATCHBOX_VECTORUTILS_HPP

#include <vector>
#include <string>

/*!
 * Convert the input string of numbers into the vector.
 * \snippet "1 28 28"
 */
template<typename T>
std::vector<T> createVectorFromString(const std::string &input_string) {
    std::vector<T> dst_vec;
    char *p_next;
    T elem;
    bool bFloat = std::is_same<T, float>::value;
    if (!bFloat) {
        elem = strtol(input_string.c_str(), &p_next, 10);
    } else {
        elem = strtof(input_string.c_str(), &p_next);
    }

    dst_vec.push_back(elem);
    while (*p_next) {
        if (!bFloat) {
            elem = strtol(p_next, &p_next, 10);
        } else {
            elem = strtof(p_next, &p_next);
        }
        dst_vec.push_back(elem);
    }
    return dst_vec;
}
#endif //MATCHBOX_VECTORUTILS_HPP
