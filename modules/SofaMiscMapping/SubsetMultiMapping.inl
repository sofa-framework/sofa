/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL

#include <sofa/core/visual/VisualParams.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <iostream>
#include <SofaBaseMechanics/IdentityMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::init()
{
    assert( indexPairs.getValue().size()%2==0 );
    const unsigned indexPairSize = indexPairs.getValue().size()/2;

    this->toModels[0]->resize( indexPairSize );

    Inherit::init();

    unsigned Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size;

    for( unsigned i=0; i<baseMatrices.size(); i++ )
        delete baseMatrices[i];

    typedef linearsolver::EigenSparseMatrix<TIn,TOut> Jacobian;
    baseMatrices.resize( this->getFrom().size() );
    helper::vector<Jacobian*> jacobians( this->getFrom().size() );
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i] = jacobians[i] = new linearsolver::EigenSparseMatrix<TIn,TOut>;
        jacobians[i]->resize(Nout*indexPairSize,Nin*this->fromModels[i]->getSize() ); // each jacobian has the same number of rows
    }

    // fill the Jacobians
    for(unsigned i=0; i<indexPairSize; i++)
    {
        unsigned parent = indexPairs.getValue()[i*2];
        Jacobian* jacobian = jacobians[parent];
        unsigned bcol = indexPairs.getValue()[i*2+1];  // parent particle
        for(unsigned k=0; k<Nout; k++ )
        {
            unsigned row = i*Nout + k;

            // the Jacobian could be filled in order, but empty rows should have to be managed
            jacobian->add( row, Nin*bcol +k, (SReal)1. );
        }
    }

    // finalize the Jacobians
    for(unsigned i=0; i<baseMatrices.size(); i++ )
        baseMatrices[i]->compress();
}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::SubsetMultiMapping()
    : Inherit()
    , indexPairs( initData( &indexPairs, helper::vector<unsigned>(), "indexPairs", "list of couples (parent index + index in the parent)"))
{}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::~SubsetMultiMapping()
{
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        delete baseMatrices[i];
    }
}

template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* SubsetMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( const core::BaseState* from, int index)
{

    // find the index of the parent state
    unsigned i;
    for(i=0; i<this->fromModels.size(); i++)
        if(this->fromModels.get(i)==from )
            break;
    if(i==this->fromModels.size())
    {
        serr<<"SubsetMultiMapping<TIn, TOut>::addPoint, parent "<<from->getName()<<" not found !"<< sendl;
        assert(0);
    }

    addPoint(i, index);
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::addPoint( int from, int index)
{
    assert((size_t)from < this->fromModels.size());
    helper::vector<unsigned>& indexPairsVector = *indexPairs.beginEdit();
    indexPairsVector.push_back(from);
    indexPairsVector.push_back(index);
    indexPairs.endEdit();
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
{
    //apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos)
    //OutVecCoord& out = *outPos[0];

    OutVecCoord& out = *(dataVecOutPos[0]->beginEdit(mparams));

    for(unsigned i=0; i<out.size(); i++)
    {
//        cerr<<"SubsetMultiMapping<TIn, TOut>::apply, i = "<< i <<", indexPair = " << indexPairs[i*2] << ", " << indexPairs[i*2+1] <<", inPos size = "<< inPos.size() <<", inPos[i] = " << (*inPos[indexPairs[i*2]]) << endl;
//        cerr<<"SubsetMultiMapping<TIn, TOut>::apply, out = "<< out << endl;
        const InDataVecCoord* inPosPtr = dataVecInPos[indexPairs.getValue()[i*2]];
        const InVecCoord& inPos = (*inPosPtr).getValue();

        //out[i] =  inPos[indexPairs.getValue()[i*2+1]];
        helper::eq( out[i], inPos[indexPairs.getValue()[i*2+1]] );
    }

    dataVecOutPos[0]->endEdit(mparams);

}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
{
    OutVecDeriv& out = *(dataVecOutVel[0]->beginEdit(mparams));

    for(unsigned i=0; i<out.size(); i++)
    {
        const InDataVecDeriv* inDerivPtr = dataVecInVel[indexPairs.getValue()[i*2]];
        const InVecDeriv& inDeriv = (*inDerivPtr).getValue();

//        out[i] = inDeriv[indexPairs.getValue()[i*2+1]];
        helper::eq( out[i], inDeriv[indexPairs.getValue()[i*2+1]] );
    }

    dataVecOutVel[0]->endEdit(mparams);
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT( const core::ConstraintParams* /*cparams*/, const helper::vector< InDataMatrixDeriv* >& dOut, const helper::vector< const OutDataMatrixDeriv* >& dIn)
{
    helper::vector<unsigned>  indexP = indexPairs.getValue();

    // hypothesis: one child only:
    const OutMatrixDeriv& in = dIn[0]->getValue();

    if (dOut.size() != this->fromModels.size())
    {
        serr<<"problem with number of output constraint matrices"<<sendl;
        return;
    }

    typename OutMatrixDeriv::RowConstIterator rowItEnd = in.end();
    // loop on the constraints defined on the child of the mapping
    for (typename OutMatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {

        typename OutMatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename OutMatrixDeriv::ColConstIterator colItEnd = rowIt.end();


        // A constraint can be shared by several nodes,
        // these nodes can be linked to 2 different parents. // TODO handle more parents
        // we need to add a line to each parent that is concerned by the constraint


        while (colIt != colItEnd)
        {
            unsigned int index_parent = indexP[colIt.index()*2]; // 0 or 1 (for now...)
            // writeLine provide an iterator on the line... if this line does not exist, the line is created:
            typename InMatrixDeriv::RowIterator o = dOut[index_parent]->beginEdit()->writeLine(rowIt.index());
            dOut[index_parent]->endEdit();

            // for each col of the constraint direction, it adds a col in the corresponding parent's constraint direction
            if(indexPairs.getValue()[colIt.index()*2+1] < (unsigned int)this->fromModels[index_parent]->getSize())
            {
                InDeriv tmp; helper::eq( tmp, colIt.val() );
                o.addCol(indexP[colIt.index()*2+1], tmp);
            }
            ++colIt;
        }



    }
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    const OutDataVecDeriv* cderData = dataVecInForce[0];
    const OutVecDeriv& cder = cderData->getValue();

    for(unsigned i=0; i<cder.size(); i++)
    {
        InDataVecDeriv* inDerivPtr = dataVecOutForce[indexPairs.getValue()[i*2]];
        InVecDeriv& inDeriv = *(*inDerivPtr).beginEdit(mparams);
        helper::peq( inDeriv[indexPairs.getValue()[i*2+1]], cder[i] );
        (*inDerivPtr).endEdit(mparams);
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
