/**
 * A DataSetIterator handles traversing through a dataset and preparing data for a neural network
 * Abstraction layer on top of mxnet DataIter to make working with data iterators easier
 */

#ifndef MATCHBOX_DATASETITERATOR_HPP
#define MATCHBOX_DATASETITERATOR_HPP

#include <mxnet-cpp/MxNetCpp.h>
#include <fileutil.hpp>

using namespace mxnet::cpp;

class DataSetIterator {

public :
    /**
    * Get data iterator
    * @return
    */
    virtual MXDataIter getMXDataIter() = 0;

    /**
     * Batch size
     * @return
     */
    virtual int batch() = 0;

    /**
     * The number of labels for the dataset
     * @return
     */
    virtual int totalOutcomes() = 0;

    /**
     * Get dataset iterator class labels, if any.
     * @return
     */
    virtual std::vector<std::string> getLabels() = 0;

    /**
     * @param file
     */
    inline void validateFile(std::string &file) {
        if (!(FileExists(file))) {
            LG << "Error:  file does not exist: " << file;
            throw std::runtime_error("Unable to create iterator, file not found " + file);
        }
    }

    /**
     * default destructor
     */
    virtual ~DataSetIterator() = default;
};


#endif //MATCHBOX_DATASETITERATOR_HPP
