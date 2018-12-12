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
#define SOFA_COMPONENT_ENGINE_EXTRUDEEDGESANDGENERATEQUADS_CPP
#include <SofaGeneralEngine/ExtrudeEdgesAndGenerateQuads.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

int ExtrudeEdgesAndGenerateQuadsClass = core::RegisterObject("This engine extrudes an edge-based curve into a quad surface patch")
        .add< ExtrudeEdgesAndGenerateQuads<Vec3Types> >(true) // default template
 //SOFA_WITH_DOUBLE
        ;

template class SOFA_GENERAL_ENGINE_API ExtrudeEdgesAndGenerateQuads<Vec3Types>;
 //SOFA_WITH_DOUBLE


} // namespace constraint

} // namespace component

} // namespace sofa


