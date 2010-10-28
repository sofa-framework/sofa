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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_INL
#define SOFA_COMPONENT_MASS_UNIFORMMASS_INL

#include <sofa/component/mass/UniformMass.h>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/DrawManager.h>
#include <iostream>
#include <string.h>




namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass()
    : mass ( initData ( &mass, MassType ( 1.0f ), "mass", "Mass of each particle" ) )
    , totalMass ( initData ( &totalMass, 0.0, "totalmass", "Sum of the particles' masses" ) )
    , filenameMass ( initData ( &filenameMass, "filename", "Rigid file to load the mass parameters" ) )
    , showCenterOfGravity ( initData ( &showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , showAxisSize ( initData ( &showAxisSize, 1.0f, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , compute_mapping_inertia ( initData ( &compute_mapping_inertia, true, "compute_mapping_inertia", "to be used if the mass is placed under a mapping" ) )
    , showInitialCenterOfGravity ( initData ( &showInitialCenterOfGravity, false, "showInitialCenterOfGravity", "display the initial center of gravity of the system" ) )
    , showX0 ( initData ( &showX0, false, "showX0", "display the rest positions" ) )
    , localRange ( initData ( &localRange, defaulttype::Vec<2,int> ( -1,-1 ), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
    , m_handleTopoChange ( initData ( &m_handleTopoChange, false, "handleTopoChange", "The mass and totalMass are recomputed on particles add/remove." ) )
{
    this->addAlias ( &totalMass, "totalMass" );
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
void UniformMass<DataTypes, MassType>::setTotalMass ( double m )
{
    this->totalMass.setValue ( m );
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::reinit()
{
    if ( this->totalMass.getValue() >0 && this->mstate!=NULL )
    {
        MassType* m = this->mass.beginEdit();
        *m = ( ( typename DataTypes::Real ) this->totalMass.getValue() / this->mstate->getX()->size() );
        this->mass.endEdit();
    }
    else
    {
        this->totalMass.setValue ( this->mstate->getX()->size() *this->mass.getValue() );
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
                    *m = ( ( typename DataTypes::Real ) this->totalMass.getValue() / this->mstate->getX()->size() );
                    this->mass.endEdit();
                }
                break;

            case core::topology::POINTSREMOVED:
                if ( m_handleTopoChange.getValue() )
                {
                    this->totalMass.setValue ( this->mstate->getX()->size() * this->mass.getValue() );
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
void UniformMass<DataTypes, MassType>::addMDx ( DataVecDeriv& vres, const DataVecDeriv& vdx, double factor , const core::MechanicalParams*)
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
        res[i] += dx[i] * m;
        //serr<<"dx[i] = "<<dx[i]<<", m = "<<m<<", dx[i] * m = "<<dx[i] * m<<sendl;
    }
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::accFromF ( DataVecDeriv& va, const DataVecDeriv& vf, const core::MechanicalParams* )
{
    helper::WriteAccessor<DataVecDeriv> a = va;
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
void UniformMass<DataTypes, MassType>::addGravityToV(core::MultiVecDerivId vid, const core::MechanicalParams* mparams)
{
    if ( this->mstate && mparams)
    {
        helper::WriteAccessor<DataVecDeriv> v = *vid[this->mstate].write();

        const SReal* g = this->getContext()->getLocalGravity().ptr();
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2] );
        Deriv hg = theGravity * ( Real ) (mparams->dt());
        if ( this->f_printLog.getValue() )
            serr << "UniformMass::addGravityToV hg = "<<theGravity<<"*"<<mparams->dt()<<"="<<hg<<sendl;
        for ( unsigned int i=0; i<v.size(); i++ )
        {
            v[i] += hg;
        }
    }
}

template <class DataTypes, class MassType>
#ifdef SOFA_SUPPORT_MOVING_FRAMES
void UniformMass<DataTypes, MassType>::addForce ( DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v , const core::MechanicalParams* )
#else
void UniformMass<DataTypes, MassType>::addForce ( DataVecDeriv& vf, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ , const core::MechanicalParams* )
#endif
{

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;

    helper::WriteAccessor<DataVecDeriv> f = vf;

    unsigned int ibegin = 0;
    unsigned int iend = f.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    // weight
    const SReal* g = this->getContext()->getLocalGravity().ptr();
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
    for ( unsigned int i=ibegin; i<iend; i++ )
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
        VecDeriv& acc =  *this->mstate->getDx();
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
double UniformMass<DataTypes, MassType>::getKineticEnergy ( const DataVecDeriv& vv, const core::MechanicalParams*  ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;

    unsigned int ibegin = 0;
    unsigned int iend = v.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    double e=0;
    const MassType& m = mass.getValue();
    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        e+= v[i]*m*v[i];
    }
    //serr<<"UniformMass<DataTypes, MassType>::getKineticEnergy = "<<e/2<<sendl;
    return e/2;
}

template <class DataTypes, class MassType>
double UniformMass<DataTypes, MassType>::getPotentialEnergy ( const DataVecCoord& vx, const core::MechanicalParams*  ) const
{
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    double e = 0;
    const MassType& m = mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
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

/// Add Mass contribution to global Matrix assembling
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMToMatrix (const sofa::core::behavior::MultiMatrixAccessor* matrix, const core::MechanicalParams *mparams)
{
    const MassType& m = mass.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    const unsigned int size = this->mstate->getSize();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactor();
    for ( unsigned int i=0; i<size; i++ )
        calc ( r.matrix, m, r.offset + N*i, mFactor);
}


template <class DataTypes, class MassType>
double UniformMass<DataTypes, MassType>::getElementMass ( unsigned int ) const
{
    return ( double ) ( mass.getValue() );
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::getElementMass ( unsigned int /* index */, defaulttype::BaseMatrix *m ) const
{
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    if ( m->rowSize() != dimension || m->colSize() != dimension ) m->resize ( dimension, dimension );

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>() ( m, mass.getValue(), 0, 1 );
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw()
{
    if ( !this->getContext()->getShowBehaviorModels() )
        return;
    helper::ReadAccessor<VecCoord> x = *this->mstate->getX();

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    //serr<<"UniformMass<DataTypes, MassType>::draw() "<<x<<sendl;


    std::vector<  Vector3 > points;
    std::vector< Vec<2,int> > indices;

    Coord gravityCenter;
    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        Vector3 p;
        p = DataTypes::getCPos(x[i]);

        points.push_back ( p );
        gravityCenter += x[i];
    }
    Vec4f color(1,1,1,1);

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
    if(this->getContext()->getShowProcessorColor())
    {
        unsigned int proc=Core::Processor::get_current()->get_pid();
        color = colorTab[proc%12];
    }
#endif
    simulation::getSimulation()->DrawUtility.drawPoints ( points, 2, color);

    if ( showCenterOfGravity.getValue() )
    {
        points.clear();
        glBegin ( GL_LINES );
        glColor4f (1,1,0,1 );
        gravityCenter /= x.size();
        for ( unsigned int i=0 ; i<Coord::spatial_dimensions ; i++ )
        {
            Coord v;
            v[i] = showAxisSize.getValue();
            helper::gl::glVertexT ( gravityCenter-v );
            helper::gl::glVertexT ( gravityCenter+v );
        }
        glEnd();
    }
}

template <class DataTypes, class MassType>
bool UniformMass<DataTypes, MassType>::addBBox ( double* minBBox, double* maxBBox )
{
    helper::ReadAccessor<VecCoord> x = *this->mstate->getX();
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        //const Coord& p = x[i];
        Real p[3] = {0.0, 0.0, 0.0};
        DataTypes::get ( p[0],p[1],p[2],x[i] );
        for ( int c=0; c<3; c++ )
        {
            if ( p[c] > maxBBox[c] ) maxBBox[c] = p[c];
            if ( p[c] < minBBox[c] ) minBBox[c] = p[c];
        }
    }
    return true;
}

template<class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::loadRigidMass ( std::string )
{
    //If the template is not rigid, we hide the Data filenameMass, to avoid confusion.
    filenameMass.setDisplayed ( false );
    this->mass.setDisplayed ( false );
}


//Specialization for rigids
#ifndef SOFA_FLOAT
template<>
void UniformMass<Rigid3dTypes, Rigid3dMass>::reinit();
template<>
void UniformMass<Rigid3dTypes, Rigid3dMass>::loadRigidMass ( std::string );
template <>
void UniformMass<Rigid3dTypes, Rigid3dMass>::draw();
template <>
void UniformMass<Rigid2dTypes, Rigid2dMass>::draw();
template <>
double UniformMass<Rigid3dTypes,Rigid3dMass>::getPotentialEnergy ( const DataVecCoord& x, const core::MechanicalParams* ) const;
template <>
double UniformMass<Rigid2dTypes,Rigid2dMass>::getPotentialEnergy ( const DataVecCoord& x, const core::MechanicalParams* ) const;
template <>
void UniformMass<Vec6dTypes,double>::draw();
#endif
#ifndef SOFA_DOUBLE
template<>
void UniformMass<Rigid3fTypes, Rigid3fMass>::reinit();
template<>
void UniformMass<Rigid3fTypes, Rigid3fMass>::loadRigidMass ( std::string );
template <>
void UniformMass<Rigid3fTypes, Rigid3fMass>::draw();
template <>
void UniformMass<Rigid2fTypes, Rigid2fMass>::draw();
template <>
double UniformMass<Rigid3fTypes,Rigid3fMass>::getPotentialEnergy ( const DataVecCoord& x, const core::MechanicalParams* ) const;
template <>
double UniformMass<Rigid2fTypes,Rigid2fMass>::getPotentialEnergy ( const DataVecCoord& x, const core::MechanicalParams* ) const;
template <>
void UniformMass<Vec6fTypes,float>::draw();
#endif



} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MASS_UNIFORMMASS_INL
