/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define CGALPLUGIN_MESHGENERATIONFROMIMAGE_CPP

#define CGAL_MESH_3_VERBOSE

#include <CGALPlugin/config.h>
#include "MeshGenerationFromImage.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(MeshGenerationFromImage)

using namespace sofa::defaulttype;
using namespace cgal;

int MeshGenerationFromImageClass = sofa::core::RegisterObject("Generate tetrahedral mesh from image")
#ifndef SOFA_FLOAT
        .add< MeshGenerationFromImage<Vec3dTypes,ImageUC> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< MeshGenerationFromImage<Vec3fTypes, ImageUC> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API cgal::MeshGenerationFromImage<Vec3dTypes, ImageUC>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
 template class SOFA_CGALPLUGIN_API cgal::MeshGenerationFromImage<Vec3fTypes, ImageUC>;
#endif //SOFA_DOUBLE
