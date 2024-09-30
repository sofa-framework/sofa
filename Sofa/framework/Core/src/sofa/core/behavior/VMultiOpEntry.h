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
#include <sofa/core/MultiVecId.h>


namespace sofa::core::behavior
{

/// The composition of a TMultiVecId and a scalar representing a scaled vector
template <VecType vtype, VecAccess vaccess>
struct ScaledMultiVecId
{
    TMultiVecId<vtype, vaccess> id;
    SReal factor{ 1._sreal};

    explicit ScaledMultiVecId(TMultiVecId<vtype, vaccess> _id, SReal _factor = 1._sreal)
        : id{_id}, factor(_factor) {}

    SOFA_ATTRIBUTE_DISABLED_VMULTIOPENTRY_LINEARCOMBINATION_ID()
    DeprecatedAndRemoved first;

    SOFA_ATTRIBUTE_DISABLED_VMULTIOPENTRY_LINEARCOMBINATION_FACTOR()
    DeprecatedAndRemoved second;
};

using ScaledConstMultiVecId = ScaledMultiVecId<V_ALL, V_READ>;

/// A linear combination represented by a list of scaled vectors
template <VecType vtype, VecAccess vaccess>
using LinearCombinationMultiVecId = type::vector< ScaledMultiVecId<vtype, vaccess> >;

template <VecType vtype, VecAccess vaccess>
LinearCombinationMultiVecId<vtype, vaccess> operator+(const ScaledMultiVecId<vtype, vaccess>& a, const ScaledMultiVecId<vtype, vaccess>& b)
{
    return {a, b};
}

template <VecType vtype, VecAccess vaccess>
LinearCombinationMultiVecId<vtype, vaccess> operator+(const ScaledMultiVecId<vtype, vaccess>& a, const LinearCombinationMultiVecId<vtype, vaccess>& b)
{
    LinearCombinationMultiVecId<vtype, vaccess> result { a };
    result.insert(result.end(), b.result.begin(), b.result.end());
    return result;
}

template <VecType vtype, VecAccess vaccess>
LinearCombinationMultiVecId<vtype, vaccess> operator+(const LinearCombinationMultiVecId<vtype, vaccess>& a, const ScaledMultiVecId<vtype, vaccess>& b)
{
    LinearCombinationMultiVecId<vtype, vaccess> result { a };
    result.push_back(b);
    return result;
}

/// Data structure describing a set of linear operation on vectors to move
/// into an output vector
/// \see vMultiOp
class VMultiOpEntry
{
public:
    using Output = MultiVecId;
    using LinearCombinationConstMultiVecId = LinearCombinationMultiVecId<V_ALL, V_READ>; //corresponds to ConstMultiVecId

    explicit VMultiOpEntry(const Output& outputId = MultiVecId::null(),
                           LinearCombinationConstMultiVecId linearCombination = {})
        : m_output(outputId)
        , m_linearCombination(std::move(linearCombination))
    {}

    VMultiOpEntry(const Output& outputId, ScaledConstMultiVecId scaledId)
        : VMultiOpEntry(outputId, LinearCombinationConstMultiVecId{std::move(scaledId)})
    {}

    [[nodiscard]] const Output& getOutput() const { return m_output; }
    [[nodiscard]] const LinearCombinationConstMultiVecId& getLinearCombination() const { return m_linearCombination; }

    Output& getOutput() { return m_output; }
    LinearCombinationConstMultiVecId& getLinearCombination() { return m_linearCombination; }

    SOFA_ATTRIBUTE_DISABLED_VMULTIOPENTRY_LINEARCOMBINATION_OUTPUT()
    DeprecatedAndRemoved first;

    SOFA_ATTRIBUTE_DISABLED_VMULTIOPENTRY_LINEARCOMBINATION_LIST()
    DeprecatedAndRemoved second;

private:
    /// The linear combination will be written in this MultiVecId
    Output m_output;

    /// Definition of the operation to perform as a linear combination
    LinearCombinationConstMultiVecId m_linearCombination;
};

using VMultiOp = type::vector< VMultiOpEntry >;
}
