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
#define SOFA_COMPONENT_ENGINE_PAIRBOXROI_CPP
#include <sofa/component/engine/select/PairBoxRoi.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ObjectFactoryTemplateDeductionRules.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine::select
{

using namespace sofa::defaulttype;

int PairBoxROIClass = core::RegisterObject("Find the primitives (vertex/edge/triangle/tetrahedron) inside a given box")
        .add< PairBoxROI<Vec3Types> >()
        .add< PairBoxROI<Vec6Types> >()
        .add< PairBoxROI<Rigid3Types> >()
        .setTemplateDeductionMethod(sofa::core::getTemplateFromMechanicalState);

template class SOFA_COMPONENT_ENGINE_SELECT_API PairBoxROI<Vec3Types>;
template class SOFA_COMPONENT_ENGINE_SELECT_API PairBoxROI<Vec6Types>;
template class SOFA_COMPONENT_ENGINE_SELECT_API PairBoxROI<Rigid3Types>;
 
} //namespace sofa::component::engine::select
