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
#define SOFA_COMPONENT_ENGINE_BOXROI_CPP
#include <SofaEngine/BoxROI.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

namespace boxroi
{


using namespace sofa::defaulttype;

int BoxROIClass = core::RegisterObject("Find the primitives (vertex/edge/triangle/quad/tetrahedron/hexahedron) inside given boxes")
        .add< BoxROI<Vec3Types> >(true) //default
        .add< BoxROI<Rigid3Types> >()
        .add< BoxROI<Vec6Types> >()
 //SOFA_WITH_DOUBLE
        ;

template class SOFA_ENGINE_API BoxROI<Vec3Types>;
template class SOFA_ENGINE_API BoxROI<Rigid3Types>;
template class SOFA_ENGINE_API BoxROI<Vec6Types>;
 // SOFA_WITH_DOUBLE

} // namespace boxroi

} // namespace constraint

} // namespace component

} // namespace sofa

