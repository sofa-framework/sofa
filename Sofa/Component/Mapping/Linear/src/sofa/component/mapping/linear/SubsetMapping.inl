/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/mapping/linear/SubsetMapping.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
SubsetMapping<TIn, TOut>::SubsetMapping()
    : Inherit()
    , f_indices( initData(&f_indices, "indices", "list of input indices"))
    , f_first( initData(&f_first, sofa::InvalidID, "first", "first index (use if indices are sequential)"))
    , f_last( initData(&f_last, sofa::InvalidID, "last", "last index (use if indices are sequential)"))
    , f_radius( initData(&f_radius, (Real)1.0e-5, "radius", "search radius to find corresponding points in case no indices are given"))
    , f_handleTopologyChange( initData(&f_handleTopologyChange, true, "handleTopologyChange", "Enable support of topological changes for indices (disable if it is linked from SubsetTopologicalMapping::pointD2S)"))
    , f_ignoreNotFound( initData(&f_ignoreNotFound, false, "ignoreNotFound", "True to ignore points that are not found in the input model, they will be treated as fixed points"))
    , f_resizeToModel( initData(&f_resizeToModel, false, "resizeToModel", "True to resize the output MechanicalState to match the size of indices"))
    , l_topology(initLink("topology", "link to the topology container"))
    , matrixJ()
    , updateJ(false)
{
}

template <class TIn, class TOut>
SubsetMapping<TIn, TOut>::~SubsetMapping()
{
}


template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::clear(Size reserve)
{
    IndexArray& indices = *f_indices.beginEdit();
    indices.clear();
    if (reserve) indices.reserve(reserve);
    f_indices.endEdit();
}

template <class TIn, class TOut>
int SubsetMapping<TIn, TOut>::addPoint(Index index)
{
    IndexArray& indices = *f_indices.beginEdit();
    const Size i = Size(indices.size());
    indices.push_back(index);
    f_indices.endEdit();
    return i;
}

template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::init()
{
    const bool ignoreNotFound = f_ignoreNotFound.getValue();
    int numnotfound = 0;
    auto inSize = this->fromModel->getSize();
    if (f_indices.getValue().empty() && f_first.getValue() != sofa::InvalidID)
    {
        IndexArray& indices = *f_indices.beginEdit();
        Index first = f_first.getValue();
        Index last = f_last.getValue();
        if (first >= inSize)
            first = 0;
        if (last >= inSize)
            last = inSize-1;
        indices.resize(last-first+1);
        for (Index i=0; i<indices.size(); ++i)
            indices[i] = first+i;
        f_indices.endEdit();
    }
    else if (f_indices.getValue().empty())
    {

        // We have to construct the correspondance index
        const InVecCoord& in   =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
        const OutVecCoord& out =this->toModel->read(core::ConstVecCoordId::position())->getValue();
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
            if(!found)
            {
                ++numnotfound;
                if(!ignoreNotFound)
                {
                    msg_error() << "Point " << i << "=" << out[i] << " not found in input model within a radius of " << rmax << ".";
                }
                indices[i] = sofa::InvalidID;
            }
        }
        f_indices.endEdit();
        if (numnotfound > 0)
        {
            msg_info() << out.size() << " points, " << out.size() - numnotfound << " found, " << numnotfound << " fixed points";
        }
    }
    else if (!ignoreNotFound)
    {
        IndexArray& indices = *f_indices.beginEdit();
        for (unsigned int i=0; i<indices.size(); ++i)
        {
            if ((unsigned)indices[i] >= inSize)
            {
                msg_error() << "incorrect index "<<indices[i]<<" (input size "<<inSize<<")";
                indices.erase(indices.begin()+i);
                --i;
            }
        }
        f_indices.endEdit();
    }
    this->Inherit::init();
    
    if (f_handleTopologyChange.getValue())
    {
        if (l_topology.empty())
        {
            msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
            l_topology.set(this->getContext()->getMeshTopologyLink());

        }

        sofa::core::topology::BaseMeshTopology* topology = l_topology.get();
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        if (topology)
        {
            // Initialize functions and parameters for topological changes
            f_indices.createTopologyHandler(topology);
        }
        else
        {
            msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name << " Set handleTopologyChange to false if topology is not needed.";
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }

    postInit();
}

template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::postInit()
{
    const IndexArray& indices = f_indices.getValue();
    this->toModel->resize(Size(indices.size()));
}

template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::apply ( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    const IndexArray& indices = f_indices.getValue();
    
    if (f_resizeToModel.getValue() || this->toModel->getSize() < indices.size())
    { 
        if (this->toModel->getSize() != indices.size()) 
        { 
            this->toModel->resize(Size(indices.size())); 
        } 
    }
    
    const InVecCoord& in = dIn.getValue();
    const OutVecCoord& out0 = this->toModel->read(core::ConstVecCoordId::restPosition())->getValue();
    OutVecCoord& out = *dOut.beginEdit();
    const auto fromSize = in.size();

    out.resize(indices.size());
    for(std::size_t i = 0; i < out.size(); ++i)
    {
        if(indices[i] < fromSize)
            out[i] = in[ indices[i] ];
        else
            out[i] = out0[i];
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    const IndexArray& indices = f_indices.getValue();

    const InVecDeriv& in = dIn.getValue();
    OutVecDeriv& out = *dOut.beginEdit();
    const std::size_t fromSize = in.size();

    out.resize(indices.size());
    for(Size i = 0; i < out.size(); ++i)
    {
        if(indices[i] < fromSize)
            out[i] = in[ indices[i] ];
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::applyJT ( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    const IndexArray& indices = f_indices.getValue();

    const OutVecDeriv& in = dIn.getValue();
    InVecDeriv& out = *dOut.beginEdit();
    const std::size_t fromSize = out.size();

    if (indices.empty())
        return;

    if (!in.empty())
    {
        out.resize(this->fromModel->getSize());
    }

    for(unsigned int i = 0; i < in.size(); ++i)
    {
        if(indices[i] < fromSize)
            out[indices[i]] += in[ i ];
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void SubsetMapping<TIn, TOut>::applyJT ( const core::ConstraintParams * /*cparams*/, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn)
{
    const IndexArray& indices = f_indices.getValue();

    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();
    const unsigned int fromSize = this->fromModel->getSize();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                if(indices[colIt.index()] < fromSize)
                    o.addCol(indices[colIt.index()], colIt.val());
                ++colIt;
            }
        }
    }
    dOut.endEdit();
}

template<class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* SubsetMapping<TIn, TOut>::getJ()
{
    if (matrixJ.get() == 0 || updateJ)
    {
        const OutVecCoord& out =this->toModel->read(core::ConstVecCoordId::position())->getValue();
        const InVecCoord& in =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
        const IndexArray& indices = f_indices.getValue();
        assert(indices.size() == out.size());
        const std::size_t fromSize = in.size();

        updateJ = false;
        if (matrixJ.get() == 0 ||
            (unsigned int)matrixJ->rowBSize() != out.size() ||
            (unsigned int)matrixJ->colBSize() != in.size())
        {
            matrixJ.reset(new MatrixType(typename MatrixType::Index(out.size() * NOut), typename MatrixType::Index(in.size() * NIn)));
        }
        else
        {
            matrixJ->clear();
        }
        for (unsigned i = 0; i < indices.size(); ++i)
        {
            if(indices[i] < fromSize)
            {
                MBloc& block = *matrixJ->wblock(i, indices[i], true);
                block.identity();
            }
        }
    }
    return matrixJ.get();
}



template <class TIn, class TOut>
const typename SubsetMapping<TIn, TOut>::js_type* SubsetMapping<TIn, TOut>::getJs()
{
    using MatrixIndex = typename MatrixType::Index;
    if( !eigen.compressedMatrix.nonZeros() || updateJ ) {
        updateJ = false;

        const IndexArray& indices = f_indices.getValue();
        const std::size_t rowsBlock = indices.size();
        const std::size_t colsBlock = this->fromModel->getSize();

        const auto rows = rowsBlock * NOut;
        const auto cols = colsBlock * NIn;

        eigen.resize(MatrixIndex(rows), MatrixIndex(cols) );

        for (std::size_t i = 0; i < indices.size(); ++i) {
            for(std::size_t j = 0; j < NOut; ++j) {
                eigen.beginRow(MatrixIndex(i*NOut+j) );
                eigen.insertBack(MatrixIndex(i*NOut+j), MatrixIndex(indices[i]*NIn+j) ,1._sreal );
            }
        }
        eigen.compress();
    }

    js.resize( 1 );
    js[0] = &eigen;
    return &js;
}

} // namespace sofa::component::mapping::linear
