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
#define SOFA_COMPONENT_MASS_DIAGONALMASS_CPP
#include <SofaBaseMechanics/DiagonalMass.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template <>
SReal DiagonalMass<Rigid3dTypes, Rigid3dMass>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
{
    SReal e = 0;
    const MassVector &masses= f_mass.getValue();
    const VecCoord& _x = x.getValue();
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<_x.size(); i++)
    {
        e -= getVCenter(theGravity) * masses[i].mass * _x[i].getCenter();
    }
    return e;
}

template <>
SReal DiagonalMass<Rigid2dTypes, Rigid2dMass>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
{
    SReal e = 0;
    const MassVector &masses= f_mass.getValue();
    const VecCoord& _x = x.getValue();
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<_x.size(); i++)
    {
        e -= getVCenter(theGravity) * masses[i].mass * _x[i].getCenter();
    }
    return e;
}

template <>
void DiagonalMass<Rigid3dTypes, Rigid3dMass>::draw(const core::visual::VisualParams* vparams)
{
    const MassVector &masses= f_mass.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for (unsigned int i=0; i<x.size(); i++)
    {
        if (masses[i].mass == 0) continue;
        const Quat& orient = x[i].getOrientation();
        //orient[3] = -orient[3];
        const RigidTypes::Vec3& center = x[i].getCenter();
        RigidTypes::Vec3 len;
        // The moment of inertia of a box is:
        //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
        //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
        //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
        // So to get lx,ly,lz back we need to do
        //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
        // Note that RigidMass inertiaMatrix is already divided by M
        double m00 = masses[i].inertiaMatrix[0][0];
        double m11 = masses[i].inertiaMatrix[1][1];
        double m22 = masses[i].inertiaMatrix[2][2];
        len[0] = sqrt(m11+m22-m00);
        len[1] = sqrt(m00+m22-m11);
        len[2] = sqrt(m00+m11-m22);

        vparams->drawTool()->drawFrame(center, orient, len*showAxisSize.getValue() );

        gravityCenter += (center * masses[i].mass);
        totalMass += masses[i].mass;
    }

    if(showCenterOfGravity.getValue())
    {
        gravityCenter /= totalMass;
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

        vparams->drawTool()->drawCross(gravityCenter, showAxisSize.getValue(), color);
    }
}

template <>
void DiagonalMass<Rigid3dTypes, Rigid3dMass>::reinit()
{
    Inherited::reinit();
}

template <>
void DiagonalMass<Rigid2dTypes, Rigid2dMass>::reinit()
{
    Inherited::reinit();
}

template <>
void DiagonalMass<Rigid3dTypes, Rigid3dMass>::init()
{
    _topology = this->getContext()->getMeshTopology();
    if (!fileMass.getValue().empty()) load(fileMass.getFullPath().c_str());
    Inherited::init();
    initTopologyHandlers();

    if (this->mstate && f_mass.getValue().size() > 0 && f_mass.getValue().size() < (unsigned)this->mstate->getSize())
    {
        MassVector &masses= *f_mass.beginEdit();
        size_t i = masses.size()-1;
        size_t n = (size_t)this->mstate->getSize();
        while (masses.size() < n)
            masses.push_back(masses[i]);
        f_mass.endEdit();
    }
}

template <>
void DiagonalMass<Rigid2dTypes, Rigid2dMass>::init()
{
    _topology = this->getContext()->getMeshTopology();
    if (!fileMass.getValue().empty()) load(fileMass.getFullPath().c_str());
    Inherited::init();
    initTopologyHandlers();

    if (this->mstate && f_mass.getValue().size() > 0 && f_mass.getValue().size() < (unsigned)this->mstate->getSize())
    {
        MassVector &masses= *f_mass.beginEdit();
        size_t i = masses.size()-1;
        size_t n = (size_t)this->mstate->getSize();
        while (masses.size() < n)
            masses.push_back(masses[i]);
        f_mass.endEdit();
    }
}

template <>
void DiagonalMass<Rigid2dTypes, Rigid2dMass>::draw(const core::visual::VisualParams* vparams)
{
    const MassVector &masses= f_mass.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        if (masses[i].mass == 0) continue;
        Vec3d len;
        len[0] = len[1] = sqrt(masses[i].inertiaMatrix);
        len[2] = 0;

        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();

        vparams->drawTool()->drawFrame(center, orient, len*showAxisSize.getValue() );
    }
}


template <>
Vector6 DiagonalMass<Vec3dTypes, double>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    const MassVector &masses = f_mass.getValue();

    defaulttype::Vec6d momentum;

    for ( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Deriv linearMomentum = v[i] * masses[i];
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

template <>
Vector6 DiagonalMass<Rigid3dTypes,Rigid3dMass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    const MassVector &masses = f_mass.getValue();

    defaulttype::Vec6d momentum;

    for ( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Rigid3dTypes::Vec3 linearMomentum = v[i].getLinear() * masses[i].mass;
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Rigid3dTypes::Vec3 angularMomentum = cross( x[i].getCenter(), linearMomentum ) + ( masses[i].inertiaMassMatrix * v[i].getAngular() );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}


#endif
#ifndef SOFA_DOUBLE
template <>
SReal DiagonalMass<Rigid3fTypes, Rigid3fMass>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
{
    SReal e = 0;
    const MassVector &masses= f_mass.getValue();
    const VecCoord& _x = x.getValue();
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<_x.size(); i++)
    {
        e -= getVCenter(theGravity) * masses[i].mass * _x[i].getCenter();
    }
    return e;
}

template <>
SReal DiagonalMass<Rigid2fTypes, Rigid2fMass>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
{
    SReal e = 0;

    const MassVector &masses= f_mass.getValue();
    const VecCoord& _x = x.getValue();
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<_x.size(); i++)
    {
        e -= getVCenter(theGravity) * masses[i].mass * _x[i].getCenter();
    }
    return e;
}





template <>
void DiagonalMass<Rigid3fTypes, Rigid3fMass>::draw(const core::visual::VisualParams* vparams)
{
    const MassVector &masses= f_mass.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for (unsigned int i=0; i<x.size(); i++)
    {
        if (masses[i].mass == 0) continue;
        const Quat& orient = x[i].getOrientation();
        //orient[3] = -orient[3];
        const RigidTypes::Vec3& center = x[i].getCenter();
        RigidTypes::Vec3 len;
        // The moment of inertia of a box is:
        //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
        //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
        //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
        // So to get lx,ly,lz back we need to do
        //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
        // Note that RigidMass inertiaMatrix is already divided by M
        double m00 = masses[i].inertiaMatrix[0][0];
        double m11 = masses[i].inertiaMatrix[1][1];
        double m22 = masses[i].inertiaMatrix[2][2];
        len[0] = sqrt(m11+m22-m00);
        len[1] = sqrt(m00+m22-m11);
        len[2] = sqrt(m00+m11-m22);

        vparams->drawTool()->drawFrame(center, orient, len*showAxisSize.getValue() );

        gravityCenter += (center * masses[i].mass);
        totalMass += masses[i].mass;
    }

    if(showCenterOfGravity.getValue())
    {
        gravityCenter /= totalMass;
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

        vparams->drawTool()->drawCross(gravityCenter, showAxisSize.getValue(), color);
    }

}
template <>
void DiagonalMass<Rigid3fTypes, Rigid3fMass>::reinit()
{
    Inherited::init();
}

template <>
void DiagonalMass<Rigid2fTypes, Rigid2fMass>::reinit()
{
    Inherited::init();
}

template <>
void DiagonalMass<Rigid3fTypes, Rigid3fMass>::init()
{
    Inherited::init();
}

template <>
void DiagonalMass<Rigid2fTypes, Rigid2fMass>::init()
{
    Inherited::init();
}


template <>
void DiagonalMass<Rigid2fTypes, Rigid2fMass>::draw(const core::visual::VisualParams* vparams)
{
    const MassVector &masses= f_mass.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        if (masses[i].mass == 0) continue;
        Vec3d len;
        len[0] = len[1] = sqrt(masses[i].inertiaMatrix);
        len[2] = 0;

        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();
        vparams->drawTool()->drawFrame(center, orient, len*showAxisSize.getValue() );
    }
}


template <>
Vector6 DiagonalMass<Vec3fTypes, float>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    const MassVector &masses = f_mass.getValue();

    Vector6 momentum;

    for ( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Deriv linearMomentum = v[i] * masses[i];
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

template <>
Vector6 DiagonalMass<Rigid3fTypes,Rigid3fMass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    const MassVector &masses = f_mass.getValue();

    Vector6 momentum;

    for ( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Rigid3fTypes::Vec3 linearMomentum = v[i].getLinear() * masses[i].mass;
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Rigid3fTypes::Vec3 angularMomentum = cross( x[i].getCenter(), linearMomentum ) + ( masses[i].inertiaMassMatrix * v[i].getAngular() );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}


#endif


SOFA_DECL_CLASS(DiagonalMass)

// Register in the Factory
int DiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< DiagonalMass<Vec3dTypes,double> >()
        .add< DiagonalMass<Vec2dTypes,double> >()
        .add< DiagonalMass<Vec1dTypes,double> >()
        .add< DiagonalMass<Rigid3dTypes,Rigid3dMass> >()
        .add< DiagonalMass<Rigid2dTypes,Rigid2dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DiagonalMass<Vec3fTypes,float> >()
        .add< DiagonalMass<Vec2fTypes,float> >()
        .add< DiagonalMass<Vec1fTypes,float> >()
        .add< DiagonalMass<Rigid3fTypes,Rigid3fMass> >()
        .add< DiagonalMass<Rigid2fTypes,Rigid2fMass> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API DiagonalMass<Vec3dTypes,double>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Vec2dTypes,double>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Vec1dTypes,double>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Rigid3dTypes,Rigid3dMass>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Rigid2dTypes,Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API DiagonalMass<Vec3fTypes,float>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Vec2fTypes,float>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Vec1fTypes,float>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Rigid3fTypes,Rigid3fMass>;
template class SOFA_BASE_MECHANICS_API DiagonalMass<Rigid2fTypes,Rigid2fMass>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

