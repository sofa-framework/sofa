/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_CPP

#include "TriangularFEMForceFieldOptim.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(TriangularFEMForceFieldOptim)


using namespace sofa::defaulttype;

// Register in the Factory
int TriangularFEMForceFieldOptimClass = core::RegisterObject("Corotational Triangular finite elements")
#ifndef SOFA_FLOAT
        .add< TriangularFEMForceFieldOptim<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangularFEMForceFieldOptim<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_SIMPLE_FEM_API TriangularFEMForceFieldOptim<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_SIMPLE_FEM_API TriangularFEMForceFieldOptim<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
