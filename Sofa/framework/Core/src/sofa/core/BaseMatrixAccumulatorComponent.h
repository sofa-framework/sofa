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
#pragma once

#include <sofa/core/config.h>
#include <sofa/type/fwd.h>
#include <sofa/core/MatrixAccumulator.h>

namespace sofa::core
{

/**
 * Matrix accumulator which is also a BaseObject. It is designed to be associated with another component.
 */
template<matrixaccumulator::Contribution c>
class BaseMatrixAccumulatorComponent : public matrixaccumulator::get_abstract_strong_type<c>, public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseMatrixAccumulatorComponent, objectmodel::BaseObject);
    using ComponentType = typename matrixaccumulator::get_component_type<c>;

    ~BaseMatrixAccumulatorComponent() override = default;

    BaseMatrixAccumulatorComponent()
    : Inherit1()
    , l_associatedComponent(initLink("component", "The local matrix is associated to this component"))
    {}

    void associateObject(ComponentType* object)
    {
        l_associatedComponent.set(object);
    }

    SingleLink<MyType, ComponentType, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_associatedComponent;
};


/**
 * Provides member typedef type for known Contribution using SFINAE
 *
 * Typedef type is an abstract strong type derived from MatrixAccumulator and BaseObject, and depending on @c
 */
template<matrixaccumulator::Contribution c>
struct get_base_object_strong
{
    using ComponentType = typename matrixaccumulator::get_component_type<c>;
    using type = BaseMatrixAccumulatorComponent<c>;
};

/// Helper alias
template<matrixaccumulator::Contribution c>
using get_base_object_strong_type = typename get_base_object_strong<c>::type;

} //namespace sofa::core::behavior
