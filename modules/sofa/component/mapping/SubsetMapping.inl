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
    , f_indices( initData(&f_indices, "indices", "list of input indices"))
    , f_first( initData(&f_first, -1, "first", "first index (use if indices are sequential)"))
    , f_last( initData(&f_last, -1, "last", "last index (use if indices are sequential)"))
{
}

template <class BaseMapping>
SubsetMapping<BaseMapping>::~SubsetMapping()
{
}


template <class BaseMapping>
void SubsetMapping<BaseMapping>::clear(int reserve)
{
    IndexArray& indices = *f_indices.beginEdit();
    indices.clear();
    if (reserve) indices.reserve(reserve);
    f_indices.endEdit();
}

template <class BaseMapping>
int SubsetMapping<BaseMapping>::addPoint(int index)
{
    IndexArray& indices = *f_indices.beginEdit();
    int i = indices.size();
    indices.push_back(index);
    f_indices.endEdit();
    return i;
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
    else if (f_indices.getValue().empty())
    {


        // We have to construct the correspondance index
        const InVecCoord& in   = *this->fromModel->getX();
        const OutVecCoord& out = *this->toModel->getX();
        IndexArray& indices = *f_indices.beginEdit();
        indices.resize(out.size());

        // searching for the first corresponding point in the 'from' model (there might be several ones).
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            bool found = false;
            for (unsigned int j = 0;  j < in.size() && !found; ++j )
            {
                if ( (out[i] - in[j]).norm() < 1.0e-5 )
                {
                    indices[i] = j;
                    found = true;
                }
            }
            if (!found)
            {
                std::cerr<<"ERROR(SubsetMapping): point "<<i<<"="<<out[i]<<" not found in input model."<<std::endl;
                indices[i] = 0;
            }
        }
        f_indices.endEdit();
    }
    else
    {
        IndexArray& indices = *f_indices.beginEdit();
        for (unsigned int i=0; i<indices.size(); ++i)
        {
            if ((unsigned)indices[i] >= inSize)
            {
                std::cerr << "ERROR(SubsetMapping): incorrect index "<<indices[i]<<" (input size "<<inSize<<")\n";
                indices.erase(indices.begin()+i);
                --i;
            }
        }
        f_indices.endEdit();
    }
    this->Inherit::init();
    postInit();
}

template <class BaseMapping>
void SubsetMapping<BaseMapping>::postInit()
{
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

template <class BaseMapping>
void SubsetMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    int offset = out.size();
    out.resize(offset+in.size());

    const IndexArray& indices = f_indices.getValue();
    for(unsigned int c = 0; c < in.size(); ++c)
    {
        for(unsigned int j=0; j<in[c].size(); j++)
        {
            const typename Out::SparseDeriv cIn = in[c][j];
            out[c+offset].push_back(In::SparseDeriv( indices[cIn.index] , (typename In::Deriv) cIn.data ));
        }
    }


}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
