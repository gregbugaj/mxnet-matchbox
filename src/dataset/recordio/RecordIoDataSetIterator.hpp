#ifndef MATCHBOX_RECORDIODATASETITERATOR_HPP
#define MATCHBOX_RECORDIODATASETITERATOR_HPP

#include <dataset/DataSetIterator.hpp>

class RecordIoDataSetIterator : public DataSetIterator {

public :
    MXDataIter getMXDataIter() override;

};


#endif //MATCHBOX_RECORDIODATASETITERATOR_HPP
