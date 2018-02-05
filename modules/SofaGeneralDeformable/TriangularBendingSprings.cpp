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
//
// C++ Implementation: TriangularBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP
#include <SofaGeneralDeformable/TriangularBendingSprings.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

// #define DEBUG_TRIANGLEBS

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;





SOFA_DECL_CLASS(TriangularBendingSprings)

// Register in the Factory
int TriangularBendingSpringsClass = core::RegisterObject("Springs added to a triangular mesh to prevent bending")
#ifndef SOFA_FLOAT
        .add< TriangularBendingSprings<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangularBendingSprings<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_DEFORMABLE_API TriangularBendingSprings<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_DEFORMABLE_API TriangularBendingSprings<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa

