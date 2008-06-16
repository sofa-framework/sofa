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
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/component/MechanicalObject.inl>

#include <sofa/simulation/common/Node.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace component
{

using namespace core::componentmodel::behavior;
using namespace defaulttype;

SOFA_DECL_CLASS(MechanicalObject)

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
#ifdef SOFA_FLOAT
        .add< MechanicalObject<Vec3fTypes> >(true) // default template
#else
        .add< MechanicalObject<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Vec2dTypes> >()
        .add< MechanicalObject<Vec1dTypes> >()
        .add< MechanicalObject<Vec6dTypes> >()
        .add< MechanicalObject<Rigid3dTypes> >()
        .add< MechanicalObject<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Vec2fTypes> >()
        .add< MechanicalObject<Vec1fTypes> >()
        .add< MechanicalObject<Vec6fTypes> >()
        .add< MechanicalObject<Rigid3fTypes> >()
        .add< MechanicalObject<Rigid2fTypes> >()
#endif
        .add< MechanicalObject<LaparoscopicRigid3Types> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
#ifndef SOFA_FLOAT
template class MechanicalObject<Vec3dTypes>;
template class MechanicalObject<Vec2dTypes>;
template class MechanicalObject<Vec1dTypes>;
template class MechanicalObject<Vec6dTypes>;
template class MechanicalObject<Rigid3dTypes>;
template class MechanicalObject<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class MechanicalObject<Vec3fTypes>;
template class MechanicalObject<Vec2fTypes>;
template class MechanicalObject<Vec1fTypes>;
template class MechanicalObject<Vec6fTypes>;
template class MechanicalObject<Rigid3fTypes>;
template class MechanicalObject<Rigid2fTypes>;
#endif
template class MechanicalObject<LaparoscopicRigid3Types>;



#ifndef SOFA_FLOAT
template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::applyRotation (const defaulttype::Quat q)
{
    VecCoord& x = *this->getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() *= q;
    }
}

template <>
bool MechanicalObject<Vec1dTypes>::addBBox(double* /*minBBox*/, double* /*maxBBox*/)
{
    return false; // ignore 1D DOFs for 3D bbox
}
#endif

#ifndef SOFA_DOUBLE
template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::applyRotation (const defaulttype::Quat q)
{
    VecCoord& x = *this->getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i].getCenter() = q.rotate(x[i].getCenter());
        x[i].getOrientation() *= q;
    }
}

template <>
bool MechanicalObject<Vec1fTypes>::addBBox(double* /*minBBox*/, double* /*maxBBox*/)
{
    return false; // ignore 1D DOFs for 3D bbox
}
#endif


} // namespace component

} // namespace sofa
