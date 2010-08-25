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
#define SOFA_COMPONENT_MASS_FRAMEDIAGONALMASS_CPP
#include <sofa/component/mass/FrameDiagonalMass.inl>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FrameDiagonalMass)

// Register in the Factory
int FrameDiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< FrameDiagonalMass<Rigid3dTypes,Frame3x6dMass> >()
        .add< FrameDiagonalMass<Affine3dTypes,Frame3x12dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FrameDiagonalMass<Rigid3fTypes,Frame3x6fMass> >()
        .add< FrameDiagonalMass<Affine3fTypes,Frame3x12fMass> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MASS_API FrameDiagonalMass<Rigid3dTypes,Frame3x6dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MASS_API FrameDiagonalMass<Rigid3fTypes,Frame3x6fMass>;
#endif


///////////////////////////////////////////////////////////////////////////////
//                     Affine3dTypes, Frame3x12dMass                         //
///////////////////////////////////////////////////////////////////////////////

template <>
void FrameDiagonalMass<Affine3dTypes, Frame3x12dMass>::addForce ( VecDeriv& f, const VecCoord& , const VecDeriv& v )
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;

    const MassVector &masses= f_mass.getValue();

    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2] );

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector ( aframe );

    // add weight and inertia force
    const double& invDt = 1./this->getContext()->getDt();
    for ( unsigned int i=0; i<masses.size(); i++ )
    {
        Deriv fDamping = - (masses[i] * v[i] * damping.getValue() * invDt);
        f[i] += theGravity*masses[i] + fDamping; //  + core::behavior::inertiaForce ( vframe,aframe,masses[i],x[i],v[i] );
    }
}

template <>
void FrameDiagonalMass<Affine3dTypes, Frame3x12dMass>::draw()
{
    const MassVector &masses= f_mass.getValue();
    if ( !this->getContext()->getShowBehaviorModels() ) return;
    VecCoord& x = *this->mstate->getX();
    if( x.size() != masses.size()) return;
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        // Deform the frame
        glPushMatrix();
        const RigidTypes::Vec3& center = x[i].getCenter();
        const Mat33& affine = x[i].getAffine();
        double glTransform[16];
        glTransform[0] = affine[0][0];
        glTransform[1] = affine[1][0];
        glTransform[2] = affine[2][0];
        glTransform[3] = 0;
        glTransform[4] = affine[0][1];
        glTransform[5] = affine[1][1];
        glTransform[6] = affine[2][1];
        glTransform[7] = 0;
        glTransform[8] = affine[0][2];
        glTransform[9] = affine[1][2];
        glTransform[10] = affine[2][2];
        glTransform[11] = 0;
        glTransform[12] = center[0];
        glTransform[13] = center[1];
        glTransform[14] = center[2];
        glTransform[15] = 1;
        glMultMatrixd( glTransform);

        simulation::getSimulation()->DrawUtility.drawFrame(Vec3(), Quat(), Vec3d(1,1,1)*showAxisSize.getValue() );

        gravityCenter += ( center * masses[i].mass );
        totalMass += masses[i].mass;
        glPopMatrix();
    }

    if ( showCenterOfGravity.getValue() )
    {
        glColor3f ( 1,1,0 );
        glBegin ( GL_LINES );
        gravityCenter /= totalMass;
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( showAxisSize.getValue(),0,0 ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( showAxisSize.getValue(),0,0 ) );
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( 0,showAxisSize.getValue(),0 ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( 0,showAxisSize.getValue(),0 ) );
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( 0,0,showAxisSize.getValue() ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( 0,0,showAxisSize.getValue() ) );
        glEnd();
    }
}

template<>
void FrameDiagonalMass<Affine3dTypes, Frame3x12dMass>::computeRelRot ( Mat33& , const Coord& , const Coord& )
{
}

///////////////////////////////////////////////////////////////////////////////
//                     Affine3fTypes, Frame3x12fMass                         //
///////////////////////////////////////////////////////////////////////////////

template <>
void FrameDiagonalMass<Affine3fTypes, Frame3x12fMass>::addForce ( VecDeriv& f, const VecCoord& , const VecDeriv& v )
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;

    const MassVector &masses= f_mass.getValue();

    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2] );

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector ( aframe );

    // add weight and inertia force
    const double& invDt = 1./this->getContext()->getDt();
    for ( unsigned int i=0; i<masses.size(); i++ )
    {
        Deriv fDamping = - (masses[i] * v[i] * damping.getValue() * invDt);
        f[i] += theGravity*masses[i] + fDamping; //  + core::behavior::inertiaForce ( vframe,aframe,masses[i],x[i],v[i] );
    }
}

template <>
void FrameDiagonalMass<Affine3fTypes, Frame3x12fMass>::draw()
{
    const MassVector &masses= f_mass.getValue();
    if ( !this->getContext()->getShowBehaviorModels() ) return;
    VecCoord& x = *this->mstate->getX();
    if( x.size() != masses.size()) return;
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        //const Mat33& affine = x[i].getAffine(); // TODO multiply gl matrix by affine bfore drawing the Frame
        const RigidTypes::Vec3& center = x[i].getCenter();

        simulation::getSimulation()->DrawUtility.drawFrame(center, Quat(), Vec3d(1,1,1)*showAxisSize.getValue() );

        gravityCenter += ( center * masses[i].mass );
        totalMass += masses[i].mass;
    }

    if ( showCenterOfGravity.getValue() )
    {
        glColor3f ( 1,1,0 );
        glBegin ( GL_LINES );
        gravityCenter /= totalMass;
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( showAxisSize.getValue(),0,0 ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( showAxisSize.getValue(),0,0 ) );
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( 0,showAxisSize.getValue(),0 ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( 0,showAxisSize.getValue(),0 ) );
        helper::gl::glVertexT ( gravityCenter - RigidTypes::Vec3 ( 0,0,showAxisSize.getValue() ) );
        helper::gl::glVertexT ( gravityCenter + RigidTypes::Vec3 ( 0,0,showAxisSize.getValue() ) );
        glEnd();
    }
}

template<>
void FrameDiagonalMass<Affine3fTypes, Frame3x12fMass>::computeRelRot ( Mat33& , const Coord& , const Coord& )
{
}


#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MASS_API FrameDiagonalMass<Affine3dTypes,Frame3x12dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MASS_API FrameDiagonalMass<Affine3fTypes,Frame3x12fMass>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

