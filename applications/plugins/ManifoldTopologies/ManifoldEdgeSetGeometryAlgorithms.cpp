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
#include "ManifoldEdgeSetGeometryAlgorithms.inl"

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(ManifoldEdgeSetGeometryAlgorithms)
int ManifoldEdgeSetGeometryAlgorithmsClass = core::RegisterObject("ManifoldEdge set geometry algorithms")
#ifdef SOFA_FLOAT
        .add< ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3fTypes> >(true) // default template
#else
        .add< ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< ManifoldEdgeSetGeometryAlgorithms<Vec2dTypes> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Vec1dTypes> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Rigid3dTypes> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ManifoldEdgeSetGeometryAlgorithms<Vec2fTypes> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Vec1fTypes> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Rigid3fTypes> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3dTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec2dTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec1dTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Rigid3dTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec2fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec1fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Rigid3fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Rigid2fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

