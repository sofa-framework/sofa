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
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_INL

#include <sofa/core/visual/VisualParams.h>
#include <SofaMiscMapping/IdentityMultiMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
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
    m_outSize = 0;
    for( unsigned i=0 ; i<this->fromModels.size() ; ++i )
        m_outSize += this->fromModels.getSize();

    this->toModels[0]->resize( m_outSize );

    Inherit::init();

    unsigned Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size;

    for( unsigned i=0; i<baseMatrices.size(); i++ )
        delete baseMatrices[i];

    baseMatrices.resize( this->getFrom().size() );
    typedef linearsolver::EigenSparseMatrix<TIn,TOut> Jacobian;
    vector<Jacobian*> jacobians( this->getFrom().size() );

    unsigned offset = 0;
    for(unsigned i=0; i<baseMatrices.size(); i++ )
    {
        baseMatrices[i] = jacobians[i] = new linearsolver::EigenSparseMatrix<TIn,TOut>;

        size_t inmatrixsize = this->fromModels[i]->getSize()*Nin;

        jacobians[i]->resize( Nout*m_outSize, inmatrixsize ); // each jacobian has the same number of rows


        // fill the jacobian
        for(unsigned j=0; j<inmatrixsize; j++ )
            jacobians[i]->insertBack( offset+j, j, (SReal)1. );
        jacobians[i]->compress();

        offset += inmatrixsize;
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
const helper::vector<sofa::defaulttype::BaseMatrix*>* IdentityMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
{
    OutVecCoord& out = *(dataVecOutPos[0]->beginEdit(mparams));

    out.resize( m_outSize );

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

    out.resize( m_outSize );

    unsigned offset = 0;
    for(unsigned i=0; i<dataVecInVel.size(); i++ )
    {
        const InVecDeriv& in = dataVecInVel[i]->getValue();

        for(unsigned int j=0; j<in.size(); j++)
        {
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

        for(unsigned int j=0; j<in.size(); j++)
        {
            helper::eq( out[j], in[offset+j]);
        }

        dataVecOutForce[i]->endEdit(mparams);

        offset += out.size();
    }


}

template <class TIn, class TOut>
void IdentityMultiMapping<TIn, TOut>::applyJT( const core::ConstraintParams* /*cparams*/, const helper::vector< InDataMatrixDeriv* >& /*dOut*/, const helper::vector< const OutDataMatrixDeriv* >& /*dIn*/)
{
    serr<<"applyJT on matrix is not implemented"<<serr;
}




} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_INL
