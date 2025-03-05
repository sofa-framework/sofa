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

#include <sofa/component/mapping/linear/SubsetMultiMapping.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/MappingHelper.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::init()
{
    assert(d_indexPairs.getValue().size() % 2 == 0 );
    const auto indexPairSize = d_indexPairs.getValue().size() / 2;

    this->toModels[0]->resize( indexPairSize );

    Inherit::init();

    static constexpr auto Nin = TIn::deriv_total_size;
    static constexpr auto Nout = TOut::deriv_total_size;


    for (unsigned i = 0; i < baseMatrices.size(); i++)
    {
        delete baseMatrices[i];
    }

    typedef linearalgebra::EigenSparseMatrix<TIn,TOut> Jacobian;
    baseMatrices.resize( this->getFrom().size() );
    type::vector<Jacobian*> jacobians( this->getFrom().size() );
    for (unsigned i = 0; i < baseMatrices.size(); i++)
    {
        baseMatrices[i] = jacobians[i] = new linearalgebra::EigenSparseMatrix<TIn,TOut>;
        jacobians[i]->resize(Nout*indexPairSize,Nin*this->fromModels[i]->getSize() ); // each jacobian has the same number of rows
    }

    const auto& indexPairsValue = d_indexPairs.getValue();

    // fill the Jacobians
    for (unsigned i = 0; i < indexPairSize; i++)
    {
        unsigned parent = indexPairsValue[i * 2];
        Jacobian* jacobian = jacobians[parent];
        unsigned bcol = indexPairsValue[i * 2 + 1];  // parent particle
        for (unsigned k = 0; k < Nout; k++)
        {
            unsigned row = i*Nout + k;

            // the Jacobian could be filled in order, but empty rows should have to be managed
            jacobian->add( row, Nin*bcol +k, 1._sreal );
        }
    }
    // finalize the Jacobians
    for (linearalgebra::BaseMatrix* baseMat : baseMatrices)
    {
        baseMat->compress();
    }
}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::SubsetMultiMapping()
    : d_indexPairs(initData(&d_indexPairs, type::vector<unsigned>(), "indexPairs",
        "list of couples (parent index + index in the parent)"))
{
    indexPairs.setOriginalData(&d_indexPairs);
}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::~SubsetMultiMapping()
{
    for (unsigned i = 0; i < baseMatrices.size(); i++)
    {
        delete baseMatrices[i];
    }
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* SubsetMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( const core::BaseState* from, int index)
{
    // find the index of the parent state
    unsigned i;
    for (i = 0; i < this->fromModels.size(); i++)
    {
        if (this->fromModels.get(i) == from)
        {
            break;
        }
    }
    if (i == this->fromModels.size())
    {
        msg_error() << "addPoint, parent " << from->getName() << " not found !";
        assert(0);
    }

    addPoint(i, index);
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( int from, int index)
{
    assert((size_t)from < this->fromModels.size());
    auto indexPairsVector = sofa::helper::getWriteAccessor(d_indexPairs);
    indexPairsVector.push_back(from);
    indexPairsVector.push_back(index);
}

template <class DataVecOut, class DataVecIn>
void apply_impl(
    const type::vector<DataVecOut*>& dataVecOut,
    const type::vector<const DataVecIn*>& dataVecIn,
    const type::vector<unsigned>& indexPairs)
{
    auto out = sofa::helper::getWriteOnlyAccessor(*dataVecOut[0]);

    for (unsigned i = 0; i < out.size(); i++)
    {
        const unsigned stateId = indexPairs[i * 2];
        const unsigned coordId = indexPairs[i * 2 + 1];

        const DataVecIn* inPosPtr = dataVecIn[stateId];
        const typename DataVecIn::value_type& inPos = inPosPtr->getValue();

        core::eq( out[i], inPos[coordId] );
    }
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams, const type::vector<DataVecCoord_t<Out>*>& dataVecOutPos, const type::vector<const DataVecCoord_t<In>*>& dataVecInPos)
{
    SOFA_UNUSED(mparams);
    apply_impl( dataVecOutPos, dataVecInPos, d_indexPairs.getValue());
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams, const type::vector<DataVecDeriv_t<Out>*>& dataVecOutVel, const type::vector<const DataVecDeriv_t<In>*>& dataVecInVel)
{
    SOFA_UNUSED(mparams);
    apply_impl( dataVecOutVel, dataVecInVel, d_indexPairs.getValue());
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT( const core::ConstraintParams* /*cparams*/, const type::vector< DataMatrixDeriv_t<In>* >& dOut, const type::vector< const DataMatrixDeriv_t<Out>* >& dIn)
{
    type::vector<unsigned>  indexP = d_indexPairs.getValue();

    // hypothesis: one child only:
    const MatrixDeriv_t<Out>& in = dIn[0]->getValue();

    if (dOut.size() != this->fromModels.size())
    {
        msg_error() << "Problem with number of output constraint matrices";
        return;
    }

    typename MatrixDeriv_t<Out>::RowConstIterator rowItEnd = in.end();
    // loop on the constraints defined on the child of the mapping
    for (typename MatrixDeriv_t<Out>::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {

        typename MatrixDeriv_t<Out>::ColConstIterator colIt = rowIt.begin();
        typename MatrixDeriv_t<Out>::ColConstIterator colItEnd = rowIt.end();


        // A constraint can be shared by several nodes,
        // these nodes can be linked to 2 different parents. // TODO handle more parents
        // we need to add a line to each parent that is concerned by the constraint


        while (colIt != colItEnd)
        {
            unsigned int index_parent = indexP[colIt.index() * 2]; // 0 or 1 (for now...)
            // writeLine provide an iterator on the line... if this line does not exist, the line is created:
            typename MatrixDeriv_t<In>::RowIterator o = dOut[index_parent]->beginEdit()->writeLine(rowIt.index());
            dOut[index_parent]->endEdit();

            // for each col of the constraint direction, it adds a col in the corresponding parent's constraint direction
            if(d_indexPairs.getValue()[colIt.index() * 2 + 1] < (unsigned int)this->fromModels[index_parent]->getSize())
            {
                Deriv_t<In> tmp;
                core::eq( tmp, colIt.val() );
                o.addCol(indexP[colIt.index() * 2 + 1], tmp);
            }
            ++colIt;
        }
    }
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams, const type::vector<DataVecDeriv_t<In>*>& dataVecOutForce, const type::vector<const DataVecDeriv_t<Out>*>& dataVecInForce)
{
    SOFA_UNUSED(mparams);

    const DataVecDeriv_t<Out>* cderData = dataVecInForce[0];
    const VecDeriv_t<Out>& cder = cderData->getValue();

    const auto& indexPairsValue = d_indexPairs.getValue();

    for (unsigned i = 0; i < cder.size(); i++)
    {
        const unsigned stateId = indexPairsValue[i * 2];
        const unsigned coordId = indexPairsValue[i * 2 + 1];

        DataVecDeriv_t<In>* inDerivPtr = dataVecOutForce[stateId];
        auto inDeriv = sofa::helper::getWriteAccessor(*inDerivPtr);
        core::peq(inDeriv[coordId], cder[i] );
    }
}

} // namespace sofa::component::mapping::linear
