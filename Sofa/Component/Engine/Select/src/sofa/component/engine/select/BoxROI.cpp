/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/engine/select/BoxROI.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactoryTemplateDeductionRules.h>

namespace sofa::component::engine::select::boxroi
{

using namespace sofa::defaulttype;

int BoxROIClass = core::RegisterObject("Find the primitives (vertex/edge/triangle/quad/tetrahedron/hexahedron) inside given boxes")
        .add< BoxROI<Vec3Types> >(true) //default
        .add< BoxROI<Vec2Types> >()
        .add< BoxROI<Vec1Types> >()
        .add< BoxROI<Rigid3Types> >()
        .add< BoxROI<Vec6Types> >()
        .setTemplateDeductionMethod(sofa::core::getTemplateFromMechanicalState);

template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<Vec3Types>;
template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<Vec2Types>;
template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<Vec1Types>;
template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<Rigid3Types>;
template class SOFA_COMPONENT_ENGINE_SELECT_API BoxROI<Vec6Types>;
 

} // namespace sofa::component::engine::select::boxroi
