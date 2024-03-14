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
#include <sofa/component/engine/select/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::engine::select
{

/**
 * This class returns the indices given a list of values.
 */
template <class T>
class IndicesFromValues : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(IndicesFromValues,T),core::DataEngine);
    typedef T Value;
    typedef sofa::type::vector<T> VecValue;

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);
    typedef sofa::type::vector<sofa::Index> VecIndex;

protected:

    IndicesFromValues();

    ~IndicesFromValues() override;
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    core::objectmodel::Data<VecValue> f_values; ///< input values
    core::objectmodel::Data<VecValue> f_global; ///< Global values, in which the input values are searched
    core::objectmodel::Data<VecIndex> f_indices; ///< Output indices of the given values, searched in global
    core::objectmodel::Data<VecIndex> f_otherIndices; ///< Output indices of the other values, (NOT the given ones) searched in global
    core::objectmodel::Data<bool> f_recursiveSearch; ///< if set to true, output are indices of the "global" data matching with one of the values

};

#if !defined(SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<std::string>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<int>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<unsigned int>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 2> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 3> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 4> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues< type::fixed_array<unsigned int, 8> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<SReal>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<type::Vec2>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<type::Vec3>;
// extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid2Types::Coord>;
// extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid2Types::Deriv>;
// extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid3Types::Coord>;
// extern template class SOFA_COMPONENT_ENGINE_SELECT_API IndicesFromValues<defaulttype::Rigid3Types::Deriv>;

#endif

} //namespace sofa::component::engine::select
