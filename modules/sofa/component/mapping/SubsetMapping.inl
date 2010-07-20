/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_SUBSETMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SUBSETMAPPING_INL

#include "SubsetMapping.h"
#include <sofa/core/topology/BaseMeshTopology.h>

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
    , f_radius( initData(&f_radius, (Real)1.0e-5, "radius", "search radius to find corresponding points in case no indices are given"))
    , matrixJ()
    , updateJ(false)
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

// Handle topological changes
template <class BaseMapping>
void SubsetMapping<BaseMapping>::handleTopologyChange(core::topology::Topology* t)
{
    core::topology::BaseMeshTopology* topoFrom = this->fromModel->getContext()->getMeshTopology();
    if (t != topoFrom) return;
    std::list<const core::topology::TopologyChange *>::const_iterator itBegin=topoFrom->beginChange();
    std::list<const core::topology::TopologyChange *>::const_iterator itEnd=topoFrom->endChange();
    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->fromModel->getX()->size());
    f_indices.endEdit();
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
            Real rmax = f_radius.getValue();
            for (unsigned int j = 0;  j < in.size(); ++j )
            {
                Real r = (Real)((out[i] - in[j]).norm());
                if ( r < rmax )
                {
                    indices[i] = j;
                    found = true;
                    rmax = r;
                }
            }
            if (!found)
            {
                serr<<"ERROR(SubsetMapping): point "<<i<<"="<<out[i]<<" not found in input model within a radius of "<<rmax<<"."<<sendl;
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
                serr << "ERROR(SubsetMapping): incorrect index "<<indices[i]<<" (input size "<<inSize<<")"<<sendl;
                //indices.getArray().erase(indices.begin()+i);
                helper::vector<unsigned int> a; indices.swap(a); a.erase(a.begin()+i); indices.swap(a);
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
    if (indices.empty()) return;
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
    for(unsigned int i = 0; i < in.size(); ++i)
    {
        OutConstraintIterator itOut;
        std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

        for (itOut=iter.first; itOut!=iter.second; itOut++)
        {
            unsigned int indexIn = itOut->first;
            OutDeriv data = (OutDeriv) itOut->second;
            out[i+offset].add( indices[indexIn] , data );
        }
    }
}

template<class BaseMapping>
const sofa::defaulttype::BaseMatrix* SubsetMapping<BaseMapping>::getJ()
{
    if (matrixJ.get() == 0 || updateJ)
    {
        const OutVecCoord& out = *this->toModel->getX();
        const InVecCoord& in = *this->fromModel->getX();
        const IndexArray& indices = f_indices.getValue();
        assert(indices.size() == out.size());

        updateJ = false;
        if (matrixJ.get() == 0 ||
            matrixJ->rowBSize() != out.size() ||
            matrixJ->colBSize() != in.size())
        {
            matrixJ.reset(new MatrixType(out.size() * NOut, in.size() * NIn));
        }
        else
        {
            matrixJ->clear();
        }
        for (unsigned i = 0; i < indices.size(); ++i)
        {
            MBloc& block = *matrixJ->wbloc(i, indices[i], true);
            block.identity();
        }
    }
    return matrixJ.get();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
