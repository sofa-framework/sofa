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
#include <sofa/core/VecId.h>
#include <sofa/type/vector.h>

namespace sofa::core
{
template<class TDataTypes>
class State;

/**
 * \brief Proxy class for accessing elements within an imaginary container that represents the
 * cumulative sum of multiple other containers. Each individual container is represented by a
 * VecId. The class maintains a list of VecIdDeriv objects, which defines the containers
 * contributing to the final cumulative sum.
 * This class provides a simplified interface for accessing elements within the cumulative
 * container. It allows retrieving specific elements using the overloaded subscript operator
 * (operator[]). When accessing an element at a particular index, the class delegates the retrieval
 * to the appropriate container represented by the associated VecIdDeriv.
 * In addition to element retrieval, the class supports dynamic management of the contributing
 * containers. It offers functions to add and remove VecId objects from the list of containers
 * that contribute to the cumulative sum.
 * \tparam TDataTypes Type of DOFs stored in the State
 */
template<class TDataTypes, VecType vtype, VecAccess vaccess>
struct AccumulationVecId
{
private:
    type::vector<TVecId<vtype, vaccess> > m_contributingVecIds{};
    const State<TDataTypes>& m_state;

public:
    using DataTypes = TDataTypes;
    using Deriv = typename DataTypes::Deriv;
    Deriv operator[](Size i) const;

    /// The provided VecDerivId container will contribute in the cumulative sum
    void addToContributingVecIds(core::ConstVecDerivId vecDerivId);

    void removeFromContributingVecIds(core::ConstVecDerivId vecDerivId);

    explicit AccumulationVecId(const State<TDataTypes>& state) : m_state(state) {}
    AccumulationVecId() = delete;
};

#if !defined(SOFA_CORE_ACCUMULATIONVECID_CPP)
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec3dTypes, V_DERIV, V_READ>;
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec2Types, V_DERIV, V_READ>;
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec1Types, V_DERIV, V_READ>;
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec6Types, V_DERIV, V_READ>;
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Rigid3Types, V_DERIV, V_READ>;
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Rigid2Types, V_DERIV, V_READ>;
extern template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec3fTypes, V_DERIV, V_READ>;
#endif

}
