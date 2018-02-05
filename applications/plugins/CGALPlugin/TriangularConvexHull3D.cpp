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
/*
 * TriangularConvexHull3D.cpp
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */
#define CGALPLUGIN_TRIANGULARCONVEXHULL3D_CPP

#include <CGALPlugin/config.h>
#include "TriangularConvexHull3D.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(TriangularConvexHull3D)

using namespace sofa::defaulttype;
using namespace cgal;

int TriangularConvexHull3DClass = sofa::core::RegisterObject("Generate triangular convex hull around points")
#ifndef SOFA_FLOAT
        .add< TriangularConvexHull3D<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< TriangularConvexHull3D<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API cgal::TriangularConvexHull3D<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API cgal::TriangularConvexHull3D<Vec3fTypes>;
#endif //SOFA_DOUBLE
