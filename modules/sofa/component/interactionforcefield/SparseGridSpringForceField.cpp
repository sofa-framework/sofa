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
#include <sofa/component/interactionforcefield/SparseGridSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

SOFA_DECL_CLASS(SparseGridSpringForceField)

using namespace sofa::defaulttype;


// Register in the Factory
int SparseGridSpringForceFieldClass = core::RegisterObject("Springs acting on the cells of a sparse grid")
#ifndef SOFA_FLOAT
        .add< SparseGridSpringForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SparseGridSpringForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_INTERACTIONFORCEFIELD_API SparseGridSpringForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_INTERACTIONFORCEFIELD_API SparseGridSpringForceField<Vec3fTypes>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

