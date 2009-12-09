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
#include <sofa/core/componentmodel/behavior/Mass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template <>
double FrameDiagonalMass<Rigid3dTypes, Frame3dMass>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    const MassVector &masses= f_mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}



template <>
void FrameDiagonalMass<Rigid3dTypes, Frame3dMass>::draw()

{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for (unsigned int i=0; i<x.size(); i++)
    {
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

        helper::gl::Axis::draw(center, orient, len*showAxisSize.getValue());

        gravityCenter += (center * masses[i].mass);
        totalMass += masses[i].mass;
    }

    if(showCenterOfGravity.getValue())
    {
        glColor3f (1,1,0);
        glBegin (GL_LINES);
        gravityCenter /= totalMass;
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        glEnd();
    }
}

template <>
void FrameDiagonalMass<Rigid3dTypes, Frame3dMass>::reinit()
{
    Inherited::reinit();
}


template <>
void FrameDiagonalMass<Rigid3dTypes, Frame3dMass>::init()
{
    Inherited::init();
}



#endif
#ifndef SOFA_DOUBLE
template <>
double FrameDiagonalMass<Rigid3fTypes, Frame3fMass>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    const MassVector &masses= f_mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}





template <>
void FrameDiagonalMass<Rigid3fTypes, Frame3fMass>::draw()

{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    Real totalMass=0;
    RigidTypes::Vec3 gravityCenter;
    for (unsigned int i=0; i<x.size(); i++)
    {
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

        helper::gl::Axis::draw(center, orient, len*showAxisSize.getValue());

        gravityCenter += (center * masses[i].mass);
        totalMass += masses[i].mass;
    }

    if(showCenterOfGravity.getValue())
    {
        glColor3f (1,1,0);
        glBegin (GL_LINES);
        gravityCenter /= totalMass;
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        glEnd();
    }
}
template <>
void FrameDiagonalMass<Rigid3fTypes, Frame3fMass>::reinit()
{
    Inherited::init();
}

template <>
void FrameDiagonalMass<Rigid3fTypes, Frame3fMass>::init()
{
    Inherited::init();
}



#endif


SOFA_DECL_CLASS(FrameDiagonalMass)

// Register in the Factory
int FrameDiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< FrameDiagonalMass<Rigid3dTypes,Frame3dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FrameDiagonalMass<Rigid3fTypes,Frame3fMass> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MASS_API FrameDiagonalMass<Rigid3dTypes,Frame3dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MASS_API FrameDiagonalMass<Rigid3fTypes,Frame3fMass>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

