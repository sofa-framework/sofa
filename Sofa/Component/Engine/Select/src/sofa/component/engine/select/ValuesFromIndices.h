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
 * This class returns the values given a list of indices.
 */
template <class T>
class ValuesFromIndices : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ValuesFromIndices,T),core::DataEngine);
    typedef T Value;
    typedef sofa::type::vector<T> VecValue;

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);
    typedef sofa::type::vector<sofa::Index> VecIndex;

protected:
    ValuesFromIndices();
    ~ValuesFromIndices() override;

public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<VecValue> f_in; ///< input values
    Data<VecIndex> f_indices; ///< Indices of the values
    Data<VecValue> f_out; ///< Output values corresponding to the indices
    Data<std::string> f_outStr; ///< Output values corresponding to the indices, converted as a string
};

#if !defined(SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<std::string>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<int>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<unsigned int>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 2> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 3> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 4> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices< type::fixed_array<unsigned int, 8> >;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<SReal>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec2>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec3>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec4>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<type::Vec6>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid2Types::Coord>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid2Types::Deriv>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid3Types::Coord>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ValuesFromIndices<defaulttype::Rigid3Types::Deriv>;

#endif

} //namespace sofa::component::engine::select
