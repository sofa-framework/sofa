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
#define SOFA_COMPONENT_ENGINE_SHAPEMATCHING_CPP

#include <sofa/component/engine/analyze/ShapeMatching.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>

namespace sofa::component::engine::analyze
{

using namespace defaulttype;

void registerShapeMatching(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Compute target positions using shape matching deformation method by Mueller et al.")
        .add< ShapeMatching<Vec3Types> >()
        .add< ShapeMatching<Rigid3Types> >());
}

template class SOFA_COMPONENT_ENGINE_ANALYZE_API ShapeMatching<Vec3Types>;
template class SOFA_COMPONENT_ENGINE_ANALYZE_API ShapeMatching<Rigid3Types>;
 


// specialization for rigids

template <>
void ShapeMatching<Rigid3Types>::doUpdate()
{
    // TO DO: shape matching for rigids as in [Muller11]
}




} //namespace sofa::component::engine::analyze
