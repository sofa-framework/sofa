#ifndef SOFA_COMPONENT_MAPPING_SUBSETMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SUBSETMAPPING_INL

#include "SubsetMapping.h"

namespace sofa
{

namespace component
{

namespace mapping
{


template <class BaseMapping>
SubsetMapping<BaseMapping>::SubsetMapping(In* from, Out* to)
    : Inherit(from, to)
    , f_indices( dataField(&f_indices, "indices", "list of input indices"))
    , f_first( dataField(&f_first, -1, "first", "first index (use if indices are sequential)"))
    , f_last( dataField(&f_last, -1, "last", "last index (use if indices are sequential)"))
{
}

template <class BaseMapping>
SubsetMapping<BaseMapping>::~SubsetMapping()
{
}

template <class BaseMapping>
void SubsetMapping<BaseMapping>::init()
{
    unsigned int inSize = this->fromModel->getX()->size();
    if (f_indices.getValue().empty() && f_first.getValue() != -1)
    {
        IndexArray& indices = *f_indices.beginEdit();
        unsigned int first = (unsigned int)f_first.getValue();
        unsigned int last = (unsigned int)f_last.getValue();
        if (first >= inSize)
            first = 0;
        if (last >= inSize)
            last = inSize-1;
        indices.resize(last-first+1);
        for (unsigned int i=0; i<indices.size(); ++i)
            indices[i] = first+i;
        f_indices.endEdit();
    }
    else
    {
        IndexArray& indices = *f_indices.beginEdit();
        for (unsigned int i=0; i<indices.size(); ++i)
        {
            if ((unsigned)indices[i] >= inSize)
            {
                std::cerr << "SubsetMapping: incorrect index "<<indices[i]<<" (input size "<<inSize<<")\n";
                indices.erase(indices.begin()+i);
                --i;
            }
        }
        f_indices.endEdit();
    }
    this->Inherit::init();
}

template <class BaseMapping>
void SubsetMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const IndexArray& indices = f_indices.getValue();
    out.resize(indices.size());
    for(unsigned int i = 0; i < out.size(); ++i)
    {
        out[i] = in[ indices[i] ];
    }
}

template <class BaseMapping>
void SubsetMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    const IndexArray& indices = f_indices.getValue();
    out.resize(indices.size());
    for(unsigned int i = 0; i < out.size(); ++i)
    {
        out[i] = in[ indices[i] ];
    }
}

template <class BaseMapping>
void SubsetMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const IndexArray& indices = f_indices.getValue();
    for(unsigned int i = 0; i < in.size(); ++i)
    {
        out[indices[i]] += in[ i ];
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
