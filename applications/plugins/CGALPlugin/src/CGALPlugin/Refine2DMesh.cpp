/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define CGALPLUGIN_REFINE2DMESH_CPP

#include <CGALPlugin/config.h>
#include <CGALPlugin/Refine2DMesh.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

using namespace sofa::defaulttype;
using namespace cgal;

int Refine2DMeshClass = sofa::core::RegisterObject("Refine 2D mesh using Delaunay triangulation")
    .add< Refine2DMesh<Vec3Types> >()
;

template class SOFA_CGALPLUGIN_API cgal::Refine2DMesh<Vec3Types>;
