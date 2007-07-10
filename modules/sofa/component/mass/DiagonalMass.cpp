/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/mass/DiagonalMass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

template <>
double DiagonalMass<Rigid3dTypes, Rigid3dMass>::getPotentialEnergy( const VecCoord& x )
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
        e += theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}

template <>
double DiagonalMass<Rigid3fTypes, Rigid3fMass>::getPotentialEnergy( const VecCoord& x )
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
        e += theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}

template <>
double DiagonalMass<Rigid2dTypes, Rigid2dMass>::getPotentialEnergy( const VecCoord& x )
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
        e += theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}

template <>
double DiagonalMass<Rigid2fTypes, Rigid2fMass>::getPotentialEnergy( const VecCoord& x )
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
        e += theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}



void MassEdgeDestroyFunction<Rigid3dTypes, Rigid3dMass>(const std::vector<unsigned int> &,
        void* , vector<RigidMass> &)
{
}

template <>
void MassEdgeCreationFunction<Rigid3dTypes, Rigid3dMass>(const std::vector<unsigned int> &,
        void* , vector<RigidMass> &)
{
}



template <>
void DiagonalMass<Rigid3dTypes, Rigid3dMass>::draw()

{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
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

        helper::gl::Axis::draw(center, orient, len);
    }
}

template <>
void DiagonalMass<Rigid3fTypes, Rigid3fMass>::draw()

{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
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

        helper::gl::Axis::draw(center, orient, len);
    }
}
template <>
void DiagonalMass<Rigid3fTypes, Rigid3fMass>::init()
{
    Inherited::init();
}
template <>
void DiagonalMass<Rigid3dTypes, Rigid3dMass>::init()
{
    Inherited::init();
}
template <>
void DiagonalMass<Rigid2fTypes, Rigid2fMass>::init()
{
    Inherited::init();
}
template <>
void DiagonalMass<Rigid2dTypes, Rigid2dMass>::init()
{
    Inherited::init();
}

template <>
void DiagonalMass<Rigid2dTypes, Rigid2dMass>::draw()
{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        Vec3d len;
        len[0] = len[1] = sqrt(masses[i].inertiaMatrix);
        len[2] = 0;

        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();
        helper::gl::Axis::draw(center, orient, len);
    }
}

template <>
void DiagonalMass<Rigid2fTypes, Rigid2fMass>::draw()
{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        Vec3d len;
        len[0] = len[1] = sqrt(masses[i].inertiaMatrix);
        len[2] = 0;

        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();
        helper::gl::Axis::draw(center, orient, len);
    }
}

SOFA_DECL_CLASS(DiagonalMass)

// Register in the Factory
int DiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
        .add< DiagonalMass<Vec3dTypes,double> >()
        .add< DiagonalMass<Vec3fTypes,float> >()
        .add< DiagonalMass<Vec2dTypes,double> >()
        .add< DiagonalMass<Vec2fTypes,float> >()
        .add< DiagonalMass<Vec1dTypes,double> >()
        .add< DiagonalMass<Vec1fTypes,float> >()
        .add< DiagonalMass<Rigid3dTypes,Rigid3dMass> >()
        .add< DiagonalMass<Rigid3fTypes,Rigid3fMass> >()
        .add< DiagonalMass<Rigid2dTypes,Rigid2dMass> >()
        .add< DiagonalMass<Rigid2fTypes,Rigid2fMass> >()
        ;

template class DiagonalMass<Vec3dTypes,double>;
template class DiagonalMass<Vec3fTypes,float>;
template class DiagonalMass<Vec2dTypes,double>;
template class DiagonalMass<Vec2fTypes,float>;
template class DiagonalMass<Vec1dTypes,double>;
template class DiagonalMass<Vec1fTypes,float>;
template class DiagonalMass<Rigid3dTypes,Rigid3dMass>;
template class DiagonalMass<Rigid3fTypes,Rigid3fMass>;
template class DiagonalMass<Rigid2dTypes,Rigid2dMass>;
template class DiagonalMass<Rigid2fTypes,Rigid2fMass>;


} // namespace mass

} // namespace component

} // namespace sofa

