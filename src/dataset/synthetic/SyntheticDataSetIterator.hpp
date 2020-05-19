#ifndef MATCHBOX_SYNTHETICDATASETITERATOR_HPP
#define MATCHBOX_SYNTHETICDATASETITERATOR_HPP

#include <dataset/DataSetIterator.hpp>

class SyntheticDataSetIterator : public DataSetIterator {

public:
    SyntheticDataSetIterator(int classes) : _classes(classes) {

    }

    MXDataIter getMXDataIter() override;

private :
    int _classes;
};


#endif //MATCHBOX_SYNTHETICDATASETITERATOR_HPP
