#ifndef MATCHBOX_RECORDIODATASETITERATOR_HPP
#define MATCHBOX_RECORDIODATASETITERATOR_HPP

#include <dataset/DataSetIterator.hpp>

/**
 * Support for creating ImageIO MXNET Iterators
 * Image formats supported TIFF ( bitonal/ color) /PNG/JPG
 * For validation set, we usually don’t shuffle the order of images.
 *
 * <code>
 * </code>
 * @ref https://mxnet.apache.org/api/faq/recordio
 * @ref https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html
 * @ref https://mxnet.apache.org/api/python/docs/api/mxnet/io/index.html#mxnet.io.ImageRecordIter
 * @ref ./cpp-package/example/inference/imagenet_inference.cpp
 * ref https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.cc
 */
class RecordIoDataSetIterator : public DataSetIterator {

public :
    RecordIoDataSetIterator(std::string path_imgrec, std::string path_imgidx, Shape data_shape, int batch_size,
                            bool shuffle, int seed) {
        validateFile(path_imgrec);
        validateFile(path_imgidx);
    }

    MXDataIter getMXDataIter() override;

private :
    int _batch_size;
    int _seed;
    int _shuffle_chunk_seed;
    int _preprocess_threads = 4;
    bool _shuffle;
    std::string _path_imgrec;
    std::string _path_imgidx;
    Shape _data_shape;

    std::vector<float> rgb_mean_;
    std::vector<float> rgb_std_;
};

#endif //MATCHBOX_RECORDIODATASETITERATOR_HPP
