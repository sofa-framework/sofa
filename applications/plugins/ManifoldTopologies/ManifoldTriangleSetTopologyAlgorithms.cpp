/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "ManifoldTriangleSetTopologyAlgorithms.inl"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
int ManifoldTriangleSetTopologyAlgorithmsClass = core::RegisterObject("ManifoldTriangle set topology algorithms")
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec3Types> >(true) // default template
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec2Types> >()
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec1Types> >()
        ;
template class ManifoldTriangleSetTopologyAlgorithms<Vec3Types>;
template class ManifoldTriangleSetTopologyAlgorithms<Vec2Types>;
template class ManifoldTriangleSetTopologyAlgorithms<Vec1Types>;

} // namespace topology

} // namespace component

} // namespace sofa

