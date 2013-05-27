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
#ifndef SOFA_EXTERNALBEHAVIORMODEL_FEMGRIDBEHAVIORMODEL_INL
#define SOFA_EXTERNALBEHAVIORMODEL_FEMGRIDBEHAVIORMODEL_INL

// plugin includes
#include "FEMGridBehaviorModel.h"




// internal model includes (here SOFA but could be any library)
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/AnimateEndEvent.h>




namespace sofa
{

namespace externalBehaviorModel
{


template <class DataTypes>
FEMGridBehaviorModel<DataTypes>::FEMGridBehaviorModel() : Inherited()
        , _youngModulus( initData( &_youngModulus, (Real)50000, "youngModulus", "Uniform Young Modulus" ) )
        , _density( initData( &_density, (Real)1, "density", "Uniform Density" ) )
        , _subdivisions( initData( &_subdivisions, (unsigned)1, "subdivisions", "nb grid subdivisions" ) )
{
}






template <class DataTypes>
void FEMGridBehaviorModel<DataTypes>::init()
{
    // parents initialization
    Inherited::init();


    /////////////////////////////
    // create internal model (here by using SOFA components but could be done by any library)
    /////////////////////////////

    m_internalNode = sofa::core::objectmodel::New< sofa::simulation::tree::GNode >();

    m_internalTopology = sofa::core::objectmodel::New< component::topology::RegularGridTopology >();
    m_internalTopology->setSize(_subdivisions.getValue()+1,_subdivisions.getValue()+1,_subdivisions.getValue()+1);

    m_internalDofs = sofa::core::objectmodel::New< Dofs >();

    const Data<typename Dofs::VecCoord>* points = this->m_exposedDofs->read(core::ConstVecCoordId::position());

    Vec3 min(0,0,0), max(-9999,-9999,-9999);
    const unsigned nbNodes = points->getValue().size();
    for( unsigned i=0; i<nbNodes ; ++i )
    {
        const Coord& p = points->getValue()[i];
        for( unsigned j=0;j<3;++j)
        {
            if( p[j]<min[j] ) min[j]=p[j];
            if( p[j]>max[j] ) max[j]=p[j];
        }
    }
    m_internalTopology->setPos( min[0], max[0], min[1], max[1], min[2], max[2] );

    m_internalForceFieldAndMass = sofa::core::objectmodel::New< component::forcefield::HexahedronFEMForceFieldAndMass<DataTypes> >();
    m_internalForceFieldAndMass->setYoungModulus( _youngModulus.getValue() );
    m_internalForceFieldAndMass->setDensity( _density.getValue() );
    m_internalForceFieldAndMass->setPoissonRatio( 0.3 );


    // to constrain certain internal dof to exposed sofa dofs. Here there are exactly at the same place, they could be interpolated
    typename component::projectiveconstraintset::FixedConstraint<DataTypes>::SPtr constraint = sofa::core::objectmodel::New< component::projectiveconstraintset::FixedConstraint<DataTypes> >();
    typename component::odesolver::EulerImplicitSolver::SPtr odesolver = sofa::core::objectmodel::New< component::odesolver::EulerImplicitSolver >();
    typename component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::SPtr cg = sofa::core::objectmodel::New< component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> >();


    m_internalNode->addObject( m_internalDofs );
    m_internalNode->addObject( m_internalTopology );
    m_internalNode->addObject( m_internalForceFieldAndMass );
    m_internalNode->addObject( constraint );
    m_internalNode->addObject( odesolver );
    m_internalNode->addObject( cg );
    m_internalNode->execute<simulation::InitVisitor>(sofa::core::ExecParams::defaultInstance());
    m_internalNode->setGravity(this->getContext()->getGravity());
    m_internalNode->setDt(this->getContext()->getDt());

    constraint->f_indices.beginEdit()->clear();

    for( unsigned x=0 ; x<2 ; ++x )
        for( unsigned y=0 ; y<2 ; ++y )
            for( unsigned z=0 ; z<2 ; ++z )
            {
                mapExposedInternalIndices[x+y*2+z*4] = x*(_subdivisions.getValue())+y*_subdivisions.getValue()*(_subdivisions.getValue()+1)+z*_subdivisions.getValue()*(_subdivisions.getValue()+1)*(_subdivisions.getValue()+1);
//                std::cerr<<x+y*2+z*4<<" "<<mapExposedInternalIndices[x+y*2+z*4]<<std::endl;
                constraint->f_indices.beginEdit()->push_back(mapExposedInternalIndices[x+y*2+z*4]);
            }

    constraint->f_indices.endEdit();


    /////////////////////////////
    /////////////////////////////
    /////////////////////////////



}



template<class DataTypes>
void FEMGridBehaviorModel<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    // at each end of simulation step
    if( dynamic_cast<simulation::AnimateEndEvent*>(event) )
    {
        DataVecCoord& internalDataX = *m_internalDofs->write(core::VecCoordId::position());
        DataVecDeriv& internalDataV = *m_internalDofs->write(core::VecDerivId::velocity());
        VecCoord& internalX = *internalDataX.beginEdit();
        VecDeriv& internalV = *internalDataV.beginEdit();

        const VecCoord& exposedX = (this->m_exposedDofs->write(core::VecCoordId::position()))->getValue();
        const VecDeriv& exposedV = (this->m_exposedDofs->write(core::VecDerivId::velocity()))->getValue();

        // impose SOFA dof states to internal dofs
        for( unsigned i=0 ; i<8 ; ++i )
        {
            internalX[mapExposedInternalIndices[i]] = exposedX[i];
            internalV[mapExposedInternalIndices[i]] = exposedV[i];
        }

        internalDataX.endEdit();
        internalDataV.endEdit();

        // start the internal model ode solver (where the sofa dof states must be constrained)
        simulation::AnimateVisitor av( core::ExecParams::defaultInstance(), m_internalNode->getDt() );
        m_internalNode->execute( av );
    }
}



template<class DataTypes>
void FEMGridBehaviorModel<DataTypes>::addForce( const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v )
{
    const VecCoord& x_in = x.getValue();
    const VecCoord& v_in = v.getValue();


    DataVecCoord& internalDataX = *m_internalDofs->write(core::VecCoordId::position());
    DataVecDeriv& internalDataV = *m_internalDofs->write(core::VecDerivId::velocity());
    DataVecDeriv& internalDataF = *m_internalDofs->write(core::VecDerivId::force());
    VecCoord& internalX = *internalDataX.beginEdit();
    VecDeriv& internalV = *internalDataV.beginEdit();
    VecDeriv& internalF = *internalDataF.beginEdit();
    internalF.resize( internalX.size() );
    internalF.fill( Deriv() );

    // copy x and v from sofa dofs to internal dofs (here pure copy, but could be an interpolation)
    for( unsigned i=0 ;i<8 ; ++i )
    {
        internalX[mapExposedInternalIndices[i]] = x_in[i];
        internalV[mapExposedInternalIndices[i]] = v_in[i];
    }


    m_internalForceFieldAndMass->addForce( mparams, internalDataF, internalDataX, internalDataV );


    internalDataX.endEdit();
    internalDataV.endEdit();


    VecDeriv& f_inout = *f.beginEdit(); // lock access to the data

    f_inout.resize( x_in.size() );


    // copy the resulting internal force to sofa dofs (here pure copy, but could be the transposed of the jacobian of the interpolation)
    for( unsigned i=0 ;i<8 ; ++i )
    {
        f_inout[i] += internalF[mapExposedInternalIndices[i]];
    }


    internalDataF.endEdit();
    f.endEdit();
}

template<class DataTypes>
void FEMGridBehaviorModel<DataTypes>::addDForce( const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
{
    const VecDeriv& dx_in = dx.getValue();

    DataVecDeriv& internalDataDX = *m_internalDofs->write(core::VecDerivId::dx());
    DataVecDeriv& internalDataDF = *m_internalDofs->write(core::VecDerivId::dforce());
    VecDeriv& internalDX = *internalDataDX.beginEdit();
    VecDeriv& internalDF = *internalDataDF.beginEdit();
    internalDX.fill( Deriv() );
    internalDF.resize( internalDX.size() );
    internalDF.fill( Deriv() );

    // copy dx from sofa dofs to internal dofs (here pure copy, but could be an interpolation)
    for( unsigned i=0 ;i<8 ; ++i )
    {
        internalDX[mapExposedInternalIndices[i]] = dx_in[i];
    }


    m_internalForceFieldAndMass->addDForce( mparams, internalDataDF, internalDataDX );


    internalDataDX.endEdit();


    VecDeriv& df_inout = *df.beginEdit(); // lock access to the data

    df_inout.resize( dx_in.size() );

    // copy the resulting df to sofa dofs (here pure copy, but could be the transposed of the jacobian of the interpolation)
    for( unsigned i=0 ;i<8 ; ++i )
    {
        df_inout[i] += internalDF[mapExposedInternalIndices[i]];
    }

    internalDataDF.endEdit();
    df.endEdit();
}


template<class DataTypes>
void FEMGridBehaviorModel<DataTypes>::addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, double factor)
{
    const VecDeriv& dx_in = dx.getValue();

    DataVecDeriv& internalDataDX = *m_internalDofs->write(core::VecDerivId::dx());
    DataVecDeriv& internalDataF = *m_internalDofs->write(core::VecDerivId::force());
    VecDeriv& internalDX = *internalDataDX.beginEdit();
    VecDeriv& internalF = *internalDataF.beginEdit();
    internalDX.fill( Deriv() );
    internalF.resize( internalDX.size() );
    internalF.fill( Deriv() );

    // copy dx from sofa dofs to internal dofs (here pure copy, but could be an interpolation)
    for( unsigned i=0 ;i<8 ; ++i )
    {
        internalDX[mapExposedInternalIndices[i]] = dx_in[i];
    }


    m_internalForceFieldAndMass->addMDx( mparams, internalDataF, internalDataDX, factor );

    internalDataDX.endEdit();


    VecDeriv& f_inout = *f.beginEdit(); // lock access to the data

    f_inout.resize( dx_in.size() );

    // copy the results to sofa dofs (here pure copy, but could be the transposed of the jacobian of the interpolation)
    for( unsigned i=0 ;i<8 ; ++i )
    {
        f_inout[i] += internalF[mapExposedInternalIndices[i]];
    }


    internalDataF.endEdit();
    f.endEdit();
}


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void FEMGridBehaviorModel<DataTypes>::draw( const core::visual::VisualParams* vparams )
{
    // debug drawing of the internal model (here calling the drawing of the hexaFEM component)
    m_internalForceFieldAndMass->draw( vparams );
}




} // namespace externalBehaviorModel

} // namespace sofa

#endif // SOFA_EXTERNALBEHAVIORMODEL_FEMGRIDBEHAVIORMODEL_INL
