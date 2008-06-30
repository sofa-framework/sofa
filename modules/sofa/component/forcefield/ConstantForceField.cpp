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
#include <sofa/component/forcefield/ConstantForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(ConstantForceField)

int ConstantForceFieldClass = core::RegisterObject("Constant forces applied to given degrees of freedom")
#ifndef SOFA_FLOAT
        .add< ConstantForceField<Vec3dTypes> >()
        .add< ConstantForceField<Vec2dTypes> >()
        .add< ConstantForceField<Vec1dTypes> >()
        .add< ConstantForceField<Vec6dTypes> >()
        .add< ConstantForceField<Rigid3dTypes> >()
        .add< ConstantForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ConstantForceField<Vec3fTypes> >()
        .add< ConstantForceField<Vec2fTypes> >()
        .add< ConstantForceField<Vec1fTypes> >()
        .add< ConstantForceField<Vec6fTypes> >()
        .add< ConstantForceField<Rigid3fTypes> >()
        .add< ConstantForceField<Rigid2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class ConstantForceField<Vec3dTypes>;
template class ConstantForceField<Vec2dTypes>;
template class ConstantForceField<Vec1dTypes>;
template class ConstantForceField<Vec6dTypes>;
template class ConstantForceField<Rigid3dTypes>;
template class ConstantForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ConstantForceField<Vec3fTypes>;
template class ConstantForceField<Vec2fTypes>;
template class ConstantForceField<Vec1fTypes>;
template class ConstantForceField<Vec6fTypes>;
template class ConstantForceField<Rigid3fTypes>;
template class ConstantForceField<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <>
double ConstantForceField<Rigid3dTypes>::getPotentialEnergy(const VecCoord& x)
{
    return 0;
}
template <>
double ConstantForceField<Rigid2dTypes>::getPotentialEnergy(const VecCoord& x)
{
    return 0;
}
#endif

#ifndef SOFA_DOUBLE
template <>
double ConstantForceField<Rigid3fTypes>::getPotentialEnergy(const VecCoord& x)
{
    return 0;
}

template <>
double ConstantForceField<Rigid2fTypes>::getPotentialEnergy(const VecCoord& x)
{
    return 0;
}
#endif
} // namespace forcefield

} // namespace component

} // namespace sofa
