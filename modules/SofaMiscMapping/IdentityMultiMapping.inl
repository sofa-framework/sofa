/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_INL

#include <sofa/core/visual/VisualParams.h>
#include <SofaMiscMapping/IdentityMultiMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::init()
{
    unsigned outSize = 0;
    for( unsigned i=0 ; i<this->fromModels.size() ; ++i )
        outSize += this->fromModels[i]->getSize();

    this->toModels[0]->resize( outSize );

    Inherit::init();

    unsigned Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size;
    static const unsigned N = std::min<unsigned>(Nin, Nout);

    for( unsigned i=0; i<baseMatrices.size(); i++ )
        delete baseMatrices[i];
    baseMatrices.resize( this->getFrom().size() );


    unsigned offset = 0;
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i] = new EigenMatrix;

        EigenMatrix& J = *static_cast<EigenMatrix*>(baseMatrices[i]);

        size_t n = this->fromModels[i]->getSize();

        J.resize( Nout*outSize, Nin*n ); // each

        J.compressedMatrix.reserve( n*N );

        for( size_t i=0 ; i<n ; ++i )
        {
            for(unsigned r = 0; r < N; ++r)
            {
                const unsigned row = Nout * (offset+i) + r;
                J.compressedMatrix.startVec( row );
                const unsigned col = Nin * i + r;
                J.compressedMatrix.insertBack( row, col ) = (OutReal)1;
            }
        }
        J.compressedMatrix.finalize();

        offset += n;
    }
}

template <class TIn, class TOut>
IdentityMultiMapping<TIn, TOut>::IdentityMultiMapping()
    : Inherit()
{}

template <class TIn, class TOut>
IdentityMultiMapping<TIn, TOut>::~IdentityMultiMapping()
{
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        delete baseMatrices[i];
    }
}



template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
{
    OutVecCoord& out = *(dataVecOutPos[0]->beginEdit(mparams));

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecInPos.size(); i++ )
    {
        const InVecCoord& inpos = dataVecInPos[i]->getValue();

        for(unsigned int j=0; j<inpos.size(); j++)
        {
            helper::eq( out[offset+j], inpos[j]);
        }
        offset += inpos.size();
    }

    dataVecOutPos[0]->endEdit(mparams);
}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
{
    OutVecDeriv& out = *(dataVecOutVel[0]->beginEdit(mparams));

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecInVel.size(); i++ )
    {
        const InVecDeriv& in = dataVecInVel[i]->getValue();

        for(unsigned int j=0; j<in.size(); j++)
        {
            if( !this->maskTo[0]->isActivated() || this->maskTo[0]->getEntry(offset+j) )
                helper::eq( out[offset+j], in[j]);
        }
        offset += in.size();
    }

    dataVecOutVel[0]->endEdit(mparams);
}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    const OutVecDeriv& in = dataVecInForce[0]->getValue();

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecOutForce.size(); i++ )
    {
        InVecDeriv& out = *dataVecOutForce[i]->beginEdit(mparams);

        for(unsigned int j=0; j<out.size(); j++)
        {
            if( this->maskTo[0]->getEntry(offset+j) )
                helper::peq( out[j], in[offset+j]);
        }

        dataVecOutForce[i]->endEdit(mparams);

        offset += out.size();
    }


}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJT( const core::ConstraintParams* /*cparams*/, const helper::vector< InDataMatrixDeriv* >& /*dOut*/, const helper::vector< const OutDataMatrixDeriv* >& /*dIn*/)
{
//    serr<<"applyJT on matrix is not implemented"<<sendl;
}


template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* IdentityMultiMapping<TIn, TOut>::getJs()
{
// it looks like it is more costly to update the Jacobian matrix than using the full, unfiltered matrix in assembly
//    size_t currentHash = this->maskTo[0]->getHash();
//    if( previousMaskHash!=currentHash )
//    {
//        previousMaskHash = currentHash;
//        unsigned offset = 0;

//        unsigned Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size;
//        static const unsigned N = std::min<unsigned>(Nin, Nout);

//        for(unsigned i=0; i<baseMatrices.size(); i++ )
//        {
//            typename EigenMatrix::CompressedMatrix& J = static_cast<EigenMatrix*>(baseMatrices[i])->compressedMatrix;

//            J.setZero();

//            size_t n = this->getFrom()[i]->getSize();

//            for( size_t k=0, kend=n ; k<kend ; ++k )
//            {
//                if( this->maskTo[0]->getEntry(offset+k) )
//                {
//                    for( size_t j=0 ; j<N ; ++j )
//                    {
//                        int row = (k+offset)*Nout+j;
//                        int col = k*Nin+j;
//                        J.insert( row, col ) = (OutReal)1;
//                    }
//                }
//            }

//            offset += n;
//        }
//    }

    return &baseMatrices;
}


template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::updateForceMask()
{
    unsigned offset = 0;
    for(size_t i=0; i<this->maskFrom.size(); i++ )
    {
        helper::StateMask& maskfrom = *this->maskFrom[i];

        for( size_t j = 0 ; j<maskfrom.size() ; ++j, ++offset )
        {
            if( this->maskTo[0]->getEntry(offset) ) maskfrom.insertEntry( j );
        }
    }
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_INL
