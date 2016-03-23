/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_INL
#define SOFA_COMPONENT_MASS_UNIFORMMASS_INL

#include <SofaBaseMechanics/UniformMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/simulation/common/Simulation.h>
#include <iostream>
#include <string.h>

#ifdef SOFA_SUPPORT_MOVING_FRAMES
#include <sofa/core/behavior/InertiaForce.h>
#endif



namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass()
    : mass ( initData ( &mass, MassType ( 1.0f ), "mass", "Mass of each particle" ) )
    , totalMass ( initData ( &totalMass, (SReal)0.0, "totalmass", "Sum of the particles' masses" ) )
    , filenameMass ( initData ( &filenameMass, "filename", "Rigid file to load the mass parameters" ) )
    , showCenterOfGravity ( initData ( &showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , showAxisSize ( initData ( &showAxisSize, 1.0f, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , compute_mapping_inertia ( initData ( &compute_mapping_inertia, false, "compute_mapping_inertia", "to be used if the mass is placed under a mapping" ) )
    , showInitialCenterOfGravity ( initData ( &showInitialCenterOfGravity, false, "showInitialCenterOfGravity", "display the initial center of gravity of the system" ) )
    , showX0 ( initData ( &showX0, false, "showX0", "display the rest positions" ) )
    , localRange ( initData ( &localRange, defaulttype::Vec<2,int> ( -1,-1 ), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
    , m_handleTopoChange ( initData ( &m_handleTopoChange, false, "handleTopoChange", "The mass and totalMass are recomputed on particles add/remove." ) )
    , d_preserveTotalMass( initData ( &d_preserveTotalMass, false, "preserveTotalMass", "Prevent totalMass from decreasing when removing particles."))
{
    this->addAlias ( &totalMass, "totalMass" );
    constructor_message();
}

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::~UniformMass()
{}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMass ( const MassType& m )
{
    this->mass.setValue ( m );
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setTotalMass ( SReal m )
{
    this->totalMass.setValue ( m );
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::reinit()
{
    if ( this->totalMass.getValue() >0 && this->mstate!=NULL )
    {
        MassType *m = this->mass.beginEdit();

        if (localRange.getValue()[0] >= 0
            && localRange.getValue()[1] > 0
            && localRange.getValue()[1] + 1 < (int)this->mstate->getSize())
        {
            *m = ( ( typename DataTypes::Real ) this->totalMass.getValue() / (localRange.getValue()[1]-localRange.getValue()[0]) );
        }
        else
        {
            *m = ( ( typename DataTypes::Real ) this->totalMass.getValue() / this->mstate->getSize() );
        }
        this->mass.endEdit();
    }
    else
    {
        this->totalMass.setValue ( this->mstate->getSize() * (Real)this->mass.getValue() );
    }
}




template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::init()
{
    loadRigidMass ( filenameMass.getFullPath() );
    if ( filenameMass.getValue().empty() ) filenameMass.setDisplayed ( false );
    this->core::behavior::Mass<DataTypes>::init();
    reinit();
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::handleTopologyChange()
{
    using core::topology::TopologyChange;

    core::topology::BaseMeshTopology *bmt = this->getContext()->getMeshTopology();

    if ( bmt != 0 )
    {
        std::list< const TopologyChange * >::const_iterator it = bmt->beginChange();
        std::list< const TopologyChange * >::const_iterator itEnd = bmt->endChange();

        while ( it != itEnd )
        {
            switch ( ( *it )->getChangeType() )
            {
            case core::topology::POINTSADDED:
                if ( m_handleTopoChange.getValue() )
                {
                    MassType* m = this->mass.beginEdit();
                    *m = ( ( typename DataTypes::Real ) this->totalMass.getValue() / this->mstate->getSize() );
                    this->mass.endEdit();
                }
                break;

            case core::topology::POINTSREMOVED:
                if ( m_handleTopoChange.getValue() )
                {
                    if (!d_preserveTotalMass.getValue())
                    {
                        this->totalMass.setValue (this->mstate->getSize() * (Real)this->mass.getValue() );
                    } else {
                        this->mass.setValue( static_cast< MassType >( ( typename DataTypes::Real ) this->totalMass.getValue() / this->mstate->getSize()) );
                    }
                }
                break;

            default:
                break;
            }

            ++it;
        }
    }
}


// -- Mass interface
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDx ( const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
    helper::WriteAccessor<DataVecDeriv> res = vres;
    helper::ReadAccessor<DataVecDeriv> dx = vdx;

    unsigned int ibegin = 0;
    unsigned int iend = dx.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    MassType m = mass.getValue();
    if ( factor != 1.0 )
        m *= ( typename DataTypes::Real ) factor;

    for ( unsigned int i=ibegin; i<iend; i++ )
    {
//        cerr<<"UniformMass<DataTypes, MassType>::addMDx, df[i] = "<<res[i]<< endl;
        res[i] += dx[i] * m;
//        cerr<<"UniformMass<DataTypes, MassType>::addMDx, dx[i] = "<<dx[i]<<", m = "<<m<<", dx[i] * m = "<<dx[i] * m<< endl;
//        cerr<<"UniformMass<DataTypes, MassType>::addMDx, df[i] = "<<res[i]<< endl;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::accFromF ( const core::MechanicalParams*, DataVecDeriv& va, const DataVecDeriv& vf )
{
    helper::WriteOnlyAccessor<DataVecDeriv> a = va;
    helper::ReadAccessor<DataVecDeriv> f = vf;

    unsigned int ibegin = 0;
    unsigned int iend = f.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    const MassType& m = mass.getValue();
    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        a[i] = f[i] / m;
        // serr<<"f[i] = "<<f[i]<<", m = "<<m<<", f[i] / m = "<<f[i] / m<<sendl;
    }
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDxToVector ( defaulttype::BaseVector * /*resVect*/, const VecDeriv* /*dx*/, SReal /*mFact*/, unsigned int& /*offset*/ )
{

}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if (mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        const SReal* g = this->getContext()->getGravity().ptr();
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2] );
        Deriv hg = theGravity * ( Real ) (mparams->dt());
        if ( this->f_printLog.getValue() )
            serr << "UniformMass::addGravityToV hg = "<<theGravity<<"*"<<mparams->dt()<<"="<<hg<<sendl;
        for ( unsigned int i=0; i<v.size(); i++ )
        {
            v[i] += hg;
        }

        d_v.endEdit();
    }
}

template <class DataTypes, class MassType>
#ifdef SOFA_SUPPORT_MAPPED_MASS
void UniformMass<DataTypes, MassType>::addForce ( const core::MechanicalParams* mparams, DataVecDeriv& vf, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
#else
#ifdef SOFA_SUPPORT_MOVING_FRAMES
void UniformMass<DataTypes, MassType>::addForce ( const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& x, const DataVecDeriv& v )
#else
void UniformMass<DataTypes, MassType>::addForce ( const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ )
#endif
#endif
{
#ifndef SOFA_SUPPORT_MOVING_FRAMES
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;
#endif

    helper::WriteAccessor<DataVecDeriv> f = vf;

    unsigned int ibegin = 0;
    unsigned int iend = f.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    // weight
    const SReal* g = this->getContext()->getGravity().ptr();
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2] );
    const MassType& m = mass.getValue();
    Deriv mg = theGravity * m;
    if ( this->f_printLog.getValue() )
        serr<<"UniformMass::addForce, mg = "<<mass<<" * "<<theGravity<<" = "<<mg<<sendl;





#ifdef SOFA_SUPPORT_MOVING_FRAMES
    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;
    //     serr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in world coordinates = "<<vframe<<sendl;
    //serr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in world coordinates = "<<aframe<<sendl;
    //     serr<<"UniformMass<DataTypes, MassType>::computeForce(), this->getContext()->getLocalToWorld() = "<<this->getContext()->getPositionInWorld()<<sendl;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector ( aframe );
    //     serr<<"UniformMass<DataTypes, MassType>::computeForce(), vFrame in local coordinates= "<<vframe<<sendl;
    //     serr<<"UniformMass<DataTypes, MassType>::computeForce(), aFrame in local coordinates= "<<aframe<<sendl;
    //     serr<<"UniformMass<DataTypes, MassType>::computeForce(), mg in local coordinates= "<<mg<<sendl;
#endif


    // add weight and inertia force
    if (this->m_separateGravity.getValue() ) for ( unsigned int i=ibegin; i<iend; i++ )
        {
#ifdef SOFA_SUPPORT_MOVING_FRAMES
            f[i] += core::behavior::inertiaForce ( vframe,aframe,m,x[i],v[i] );
#endif
            //serr<<"UniformMass<DataTypes, MassType>::computeForce(), vframe = "<<vframe<<", aframe = "<<aframe<<", x = "<<x[i]<<", v = "<<v[i]<<sendl;
            //serr<<"UniformMass<DataTypes, MassType>::computeForce() = "<<mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i])<<sendl;
        }
    else for ( unsigned int i=ibegin; i<iend; i++ )
        {
#ifdef SOFA_SUPPORT_MOVING_FRAMES
            f[i] += mg + core::behavior::inertiaForce ( vframe,aframe,m,x[i],v[i] );
#else
            f[i] += mg;
#endif
            //serr<<"UniformMass<DataTypes, MassType>::computeForce(), vframe = "<<vframe<<", aframe = "<<aframe<<", x = "<<x[i]<<", v = "<<v[i]<<sendl;
            //serr<<"UniformMass<DataTypes, MassType>::computeForce() = "<<mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i])<<sendl;
        }

#ifdef SOFA_SUPPORT_MAPPED_MASS
    if ( compute_mapping_inertia.getValue() )
    {
        helper::ReadAccessor< Data<VecDeriv> > acc = *mparams->readDx(this->mstate);
        // add inertia force due to acceleration from the motion of the mapping (coriolis type force)
        if ( acc.size() != f.size() )
            return;

        for ( unsigned int i=0; i<f.size(); i++ )
        {
            Deriv coriolis = -acc[i]*m;
            f[i] += coriolis;
        }
    }
#endif
}

template <class DataTypes, class MassType>
SReal UniformMass<DataTypes, MassType>::getKineticEnergy ( const core::MechanicalParams*, const DataVecDeriv& vv  ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;

    unsigned int ibegin = 0;
    unsigned int iend = v.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    SReal e=0;
    const MassType& m = mass.getValue();
    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        e+= v[i]*m*v[i];
    }
    //serr<<"UniformMass<DataTypes, MassType>::getKineticEnergy = "<<e/2<<sendl;
    return e/2;
}

template <class DataTypes, class MassType>
SReal UniformMass<DataTypes, MassType>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& vx  ) const
{
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    SReal e = 0;
    const MassType& m = mass.getValue();
    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2] );
    Deriv mg = theGravity * m;
    //serr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, theGravity = "<<theGravity<<sendl;
    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        /*        serr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, mass = "<<mass<<sendl;
        serr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, x = "<<x[i]<<sendl;
        serr<<"UniformMass<DataTypes, MassType>::getPotentialEnergy, remove "<<theGravity*mass*x[i]<<sendl;*/
        e -= mg*x[i];
    }
    return e;
}

// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
defaulttype::Vector6
UniformMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return defaulttype::Vector6();
}

/// Add Mass contribution to global Matrix assembling
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMToMatrix (const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const MassType& m = mass.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    const unsigned int size = this->mstate->getSize();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
    unsigned int ibegin = 0;
    unsigned int iend = size;

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    for (unsigned int i=ibegin; i<iend; i++ )
        calc ( r.matrix, m, r.offset + N*i, mFactor);
}


template <class DataTypes, class MassType>
SReal UniformMass<DataTypes, MassType>::getElementMass ( unsigned int ) const
{
    return ( SReal ) ( mass.getValue() );
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::getElementMass ( unsigned int /* index */, defaulttype::BaseMatrix *m ) const
{
    static const defaulttype::BaseMatrix::Index dimension = (defaulttype::BaseMatrix::Index) defaulttype::DataTypeInfo<Deriv>::size();
    if ( m->rowSize() != dimension || m->colSize() != dimension ) m->resize ( dimension, dimension );

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>() ( m, mass.getValue(), 0, 1 );
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowBehaviorModels() )
        return;
    helper::ReadAccessor<VecCoord> x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    //serr<<"UniformMass<DataTypes, MassType>::draw() "<<x<<sendl;


    std::vector<  sofa::defaulttype::Vector3 > points;
//    std::vector<  sofa::defaulttype::Vec<2,int> > indices;

    Coord gravityCenter;
    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        sofa::defaulttype::Vector3 p;
        p = DataTypes::getCPos(x[i]);

        points.push_back ( p );
        gravityCenter += x[i];
    }
//    sofa::defaulttype::Vec4f color(1,1,1,1);

#ifdef SOFA_SMP
    static float colorTab[][4]=
    {
        {1.0f,0.0f,0.0f,1.0f},
        {1.0f,1.0f,0.0f,1.0f},
        {0.0f,1.0f,0.0f,1.0f},
        {0.0f,1.0f,1.0f,1.0f},
        {0.0f,0.0f,1.0f,1.0f},
        {0.5f,.5f,.5f,1.0f},
        {0.5f,0.0f,0.0f,1.0f},
        {.5f,.5f,0.0f,1.0f},
        {0.0f,1.0f,0.0f,1.0f},
        {0.0f,1.0f,1.0f,1.0f},
        {0.0f,0.0f,1.0f,1.0f},
        {0.5f,.5f,.5f,1.0f}
    };
    if(vparams->displayFlags().getShowProcessorColor())
    {
        unsigned int proc=Core::Processor::get_current()->get_pid();
        color = colorTab[proc%12];
    }
#endif
//    vparams->drawTool()->drawPoints ( points, 2, color);

    if ( showCenterOfGravity.getValue() )
    {
        gravityCenter /= x.size();
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

        Real axisSize = showAxisSize.getValue();
        sofa::defaulttype::Vector3 temp;

        for ( unsigned int i=0 ; i<3 ; i++ )
            if(i < Coord::spatial_dimensions )
                temp[i] = gravityCenter[i];

        vparams->drawTool()->drawCross(temp, axisSize, color);
    }
}

template<class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::loadRigidMass ( std::string )
{
    //If the template is not rigid, we hide the Data filenameMass, to avoid confusion.
    filenameMass.setDisplayed ( false );
    this->mass.setDisplayed ( false );
}

} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MASS_UNIFORMMASS_INL
