#ifndef MATCHBOX_RECORDIODATASETITERATOR_HPP
#define MATCHBOX_RECORDIODATASETITERATOR_HPP

#include <dataset/DataSetIterator.hpp>
#include <vectorutils.hpp>

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
 * @ref https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.cc
 * @ref https://github.com/KeyKy/mobilenet-mxnet/blob/master/score.py
 */
class RecordIoDataSetIterator : public DataSetIterator {

public :
    RecordIoDataSetIterator(std::string path_imgrec, std::string path_imgidx, Shape data_shape, int batch_size,
                            bool shuffle, int seed, std::string &rgb_mean, std::string &rgb_std) :
            path_imgrec_(path_imgrec), path_imgidx_(path_imgidx),
            data_shape_(data_shape), batch_size_(batch_size),
            shuffle_(shuffle), seed_(seed) {
        validateFile(path_imgrec);
        validateFile(path_imgidx);

        rgb_mean_ = createVectorFromString<float>(rgb_mean);
        rgb_std_ = createVectorFromString<float>(rgb_std);
    }

    MXDataIter getMXDataIter() override;

private :
    int batch_size_;
    int seed_;
    int shuffle_chunk_seed_;
    int preprocess_threads_ = 4;
    bool shuffle_;
    std::string path_imgrec_;
    std::string path_imgidx_;
    Shape data_shape_;
    std::vector<float> rgb_mean_;
    std::vector<float> rgb_std_;
};

#endif //MATCHBOX_RECORDIODATASETITERATOR_HPP
