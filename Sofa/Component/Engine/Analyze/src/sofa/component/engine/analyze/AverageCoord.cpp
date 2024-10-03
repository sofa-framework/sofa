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
#define SOFA_COMPONENT_ENGINE_AverageCoord_CPP
#include <sofa/component/engine/analyze/AverageCoord.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::analyze
{

using namespace sofa::defaulttype;

void registerAverageCoord(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Compute the average of coordinates.")
        .add< AverageCoord<Vec2Types> >()
        .add< AverageCoord<Vec3Types> >()
        .add< AverageCoord<Rigid2Types> >()
        .add< AverageCoord<Rigid3Types> >());
}

template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<Vec2Types>;
template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<Rigid2Types>;
template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<Vec3Types>;
template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<Rigid3Types>;
 


} //namespace sofa::component::engine::analyze
