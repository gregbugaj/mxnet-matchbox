#ifndef MATCHBOX_RECORDIODATASETITERATOR_HPP
#define MATCHBOX_RECORDIODATASETITERATOR_HPP

#include <dataiter/DataSetIterator.hpp>

class RecordIoDataSetIterator : public DataSetIterator {

public :
    MXDataIter getMXDataIter() override;

};


#endif //MATCHBOX_RECORDIODATASETITERATOR_HPP
