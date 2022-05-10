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

#include <sofa/component/mapping/linear/IdentityMultiMapping.h>
#include <sofa/core/MappingHelper.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::init()
{
    unsigned outSize = 0;
    for( unsigned i=0 ; i<this->fromModels.size() ; ++i )
        outSize += this->fromModels[i]->getSize();

    this->toModels[0]->resize( outSize );

    Inherit::init();

    static constexpr auto Nin = TIn::deriv_total_size;
    static constexpr auto Nout = TOut::deriv_total_size;
    static constexpr auto N = std::min(Nin, Nout);

    for( unsigned i=0; i<baseMatrices.size(); i++ )
        delete baseMatrices[i];
    baseMatrices.resize( this->getFrom().size() );


    unsigned offset = 0;
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i] = new EigenMatrix;

        EigenMatrix& J = *static_cast<EigenMatrix*>(baseMatrices[i]);

        const auto n = this->fromModels[i]->getSize();

        J.resize( Nout*outSize, Nin*n ); // each

        J.compressedMatrix.reserve( n*N );

        for( size_t j=0 ; j<n ; ++j )
        {
            for(unsigned r = 0; r < N; ++r)
            {
                const unsigned row = Nout * (offset+j) + r;
                J.compressedMatrix.startVec( row );
                const unsigned col = Nin * j + r;
                J.compressedMatrix.insertBack( row, col ) = static_cast<OutReal>(1);
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
void IdentityMultiMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos)
{
    SOFA_UNUSED(mparams);

    OutVecCoord& out = *(dataVecOutPos[0]->beginEdit());

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecInPos.size(); i++ )
    {
        const InVecCoord& inpos = dataVecInPos[i]->getValue();

        for(unsigned int j=0; j<inpos.size(); j++)
        {
            core::eq( out[offset+j], inpos[j]);
        }
        offset += inpos.size();
    }

    dataVecOutPos[0]->endEdit();
}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel)
{
    SOFA_UNUSED(mparams);

    OutVecDeriv& out = *(dataVecOutVel[0]->beginEdit());

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecInVel.size(); i++ )
    {
        const InVecDeriv& in = dataVecInVel[i]->getValue();

        for(unsigned int j=0; j<in.size(); j++)
        {
            core::eq( out[offset+j], in[j]);
        }
        offset += in.size();
    }

    dataVecOutVel[0]->endEdit();
}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    SOFA_UNUSED(mparams);

    const OutVecDeriv& in = dataVecInForce[0]->getValue();

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecOutForce.size(); i++ )
    {
        InVecDeriv& out = *dataVecOutForce[i]->beginEdit();

        for(unsigned int j=0; j<out.size(); j++)
        {
            core::peq( out[j], in[offset+j]);
        }

        dataVecOutForce[i]->endEdit();

        offset += out.size();
    }


}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJT( const core::ConstraintParams* /*cparams*/, const type::vector< InDataMatrixDeriv* >& /*dOut*/, const type::vector< const OutDataMatrixDeriv* >& /*dIn*/)
{

}


template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* IdentityMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

} // namespace sofa::component::mapping::linear
