/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL

#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#ifdef SOFA_HAVE_EIGEN2
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#endif
#include <sofa/core/MultiMapping.inl>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::core;


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::init()
{

    Inherit::init();

    assert( indexPairs.getValue().size()%2==0 );
    const unsigned indexPairSize = indexPairs.getValue().size()/2;

    this->toModels[0]->resize( indexPairSize );

#ifdef SOFA_HAVE_EIGEN2
    unsigned Nin = TIn::deriv_total_size, Nout = Nin;

    for( unsigned i=0; i<baseMatrices.size(); i++ )
        delete baseMatrices[i];

    typedef linearsolver::EigenSparseMatrix<TIn,TOut> Jacobian;
    baseMatrices.resize( this->getFrom().size() );
    vector<Jacobian*> jacobians( this->getFrom().size() );
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i] = jacobians[i] = new linearsolver::EigenSparseMatrix<TIn,TOut>;
        jacobians[i]->resize(Nout*indexPairSize,Nin*this->fromModels[i]->readPositions().size() ); // each jacobian has the same number of rows
    }

    // fill the jacobians
    for(unsigned i=0; i<indexPairSize; i++)
    {
        unsigned parent = indexPairs.getValue()[i*2];
        Jacobian* jacobian = jacobians[parent];
        unsigned bcol = indexPairs.getValue()[i*2+1];  // parent particle
        for(unsigned k=0; k<Nin; k++ )
        {
            unsigned row = i*Nout + k;
            jacobian->insertBack( row, Nin*bcol +k, (SReal)1. );
        }
    }
//    // fill the jacobians
//    vector<unsigned> rowIndex(this->getFrom().size(),0); // current block row index in each jacobian
//    for(unsigned i=0; i<indexPairSize; i++)
//    {
//        unsigned parent = indexPairs[i*2];
//        Jacobian* jacobian = jacobians[parent];
//        unsigned& brow = rowIndex[parent];
//        unsigned bcol = indexPairs[i*2+1];  // parent particle
//        for(unsigned k=0; k<Nin; k++ )
//        {
////            baseMatrices[ indexPairs[i*2] ]->set( Nout*i+k, Nin*indexPairs[i*2+1], (SReal)1. );
//            jacobian->beginRow(Nout*brow+k);
//            jacobian->insertBack( Nout*brow+k, Nin*bcol +k, (SReal)1. );
//        }
//        brow++;
//    }

    // finalize the jacobians
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i]->compress();
    }
#endif
}

template <class TIn, class TOut>
SubsetMultiMapping<TIn, TOut>::~SubsetMultiMapping()
{
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        delete baseMatrices[i];
    }
}

#ifdef SOFA_HAVE_EIGEN2
template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* SubsetMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}
#endif


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


    vector<unsigned>& indexPairsVector = *indexPairs.beginEdit();
    indexPairsVector.push_back(i);
    indexPairsVector.push_back(index);
    indexPairs.endEdit();

}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos)
{
    OutVecCoord& out = *outPos[0];
    out.resize(indexPairs.getValue().size()/2);
    for(unsigned i=0; i<out.size(); i++)
    {
//        cerr<<"SubsetMultiMapping<TIn, TOut>::apply, i = "<< i <<", indexPair = " << indexPairs[i*2] << ", " << indexPairs[i*2+1] <<", inPos size = "<< inPos.size() <<", inPos[i] = " << (*inPos[indexPairs[i*2]]) << endl;
//        cerr<<"SubsetMultiMapping<TIn, TOut>::apply, out = "<< out << endl;
        out[i] =
                (*inPos[indexPairs.getValue()[i*2]])
                [indexPairs.getValue()[i*2+1]];
    }

}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJ(const helper::vector< typename SubsetMultiMapping<TIn, TOut>::OutVecDeriv*>& outDeriv, const helper::vector<const InVecDeriv*>& inDeriv)
{
    OutVecDeriv& out = *outDeriv[0];
    out.resize(indexPairs.getValue().size()/2);
    for(unsigned i=0; i<out.size(); i++)
    {
        out[i] = (*inDeriv[indexPairs.getValue()[i*2]])[indexPairs.getValue()[i*2+1]];
    }
}

template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT( const helper::vector<InMatrixDeriv* >& , const helper::vector<OutMatrixDeriv* >& )
{
    cerr<<"WARNING ! SubsetMultiMapping<TIn, TOut>::applyJT not implemented for MatrixDeriv ! " << endl;
}


template <class TIn, class TOut>
void SubsetMultiMapping<TIn, TOut>::applyJT(const helper::vector<typename SubsetMultiMapping<TIn, TOut>::InVecDeriv*>& parentDeriv , const helper::vector<const OutVecDeriv*>& childDeriv )
{
    const InVecDeriv& cder = *childDeriv[0];
    for(unsigned i=0; i<cder.size(); i++)
    {
        (*parentDeriv[indexPairs.getValue()[i*2]])[indexPairs.getValue()[i*2+1]] += cder[i];
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_INL
