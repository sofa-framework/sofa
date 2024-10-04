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
#define SOFA_COMPONENT_ENGINE_DifferenceEngine_CPP
#include <sofa/component/engine/transform/DifferenceEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::engine::transform
{
    
using namespace sofa::type;
using namespace sofa::defaulttype;

void registerDifferenceEngine(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Computing the difference between two vector of dofs.")
        .add< DifferenceEngine<Vec1> >()
        .add< DifferenceEngine<Vec3> >(true));
}

template class SOFA_COMPONENT_ENGINE_TRANSFORM_API DifferenceEngine<Vec1>;
template class SOFA_COMPONENT_ENGINE_TRANSFORM_API DifferenceEngine<Vec3>;

} //namespace sofa::component::engine::transform
