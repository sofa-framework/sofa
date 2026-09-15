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

#include <sofa/type/config.h>

#include <iostream>
#include <algorithm>

namespace sofa::type
{

/**
 * @class ClampedScalar
 * @brief A templated class that ensures a stored scalar value never falls outside
 *        a defined minimum and maximum range [minBound, maxBound].
 *
 * The bounds are set during construction and cannot be changed afterward.
 *
 * @tparam T The underlying numeric type (e.g., float, double).
 */
template <typename T>
class ClampedScalar
{
private:
    T m_value;        // The current stored value, guaranteed to be within bounds.
    T m_minBound;     // The absolute minimum allowed value (immutable after construction).
    T m_maxBound;     // The absolute maximum allowed value (immutable after construction).

public:
    /**
     * @brief Constructor initializes the scalar with defined bounds and an initial value.
     *
     * Note: The constructor automatically clamps the provided initial value against
     *       the given bounds.
     *
     * @param initialValue The starting value (will be clamped if out of range).
     * @param bounds A pair defining {{min_val, max_val}}. The actual min/max are derived from this pair.
     */
    constexpr ClampedScalar(T initialValue = {}, const std::pair<T,T> bounds = {{}, static_cast<T>(1)})
        : m_minBound(std::min(bounds.first, bounds.second)), m_maxBound(std::max(bounds.first, bounds.second))
    {
        // Use the internal setter to ensure the initial value is correctly clamped
        setValue(initialValue);
    }

    constexpr T getMinBound() const
    {
        return m_minBound;
    }

    constexpr T getMaxBound() const
    {
        return m_maxBound;
    }

    /**
     * @brief Sets the internal value after clamping it against the established bounds [minBound, maxBound].
     *
     * This method is used internally by constructors and setters.
     *
     * @param rawValue The raw input value (may be outside bounds).
     */
    constexpr void setValue(T rawValue)
    {
        m_value = std::clamp(rawValue, m_minBound, m_maxBound);
    }

    /**
     * @brief Returns the current clamped value.
     * @return The stored value, guaranteed to be within [min_bound_, max_bound_].
     */
    constexpr T getValue() const
    {
        return m_value;
    }

    /**
     * @brief Calculates what the clamped result of a given input would be
     *        without modifying the object's state. Useful for prediction/calculation.
     * @param rawValue The unconstrained input value.
     * @return The clamped version of rawValue.
     */
    constexpr T calculateClamped(T rawValue) const
    {
        return std::clamp(rawValue, m_minBound, m_maxBound);
    }

    /**
     * @brief Conversion operator allows the ClampedScalar to be implicitly treated as its underlying type T.
     * @return The stored, clamped value (T).
     */
    constexpr operator T() const
    {
        return m_value;
    }

    /**
     * @brief Overloads assignment with a raw scalar value for concise state updates.
     *
     * This operation clamps the new value against the existing bounds and updates the internal state.
     *
     * @param newValue The raw input value to assign.
     * @return Reference to this object.
     */
    constexpr ClampedScalar<T>& operator=(T newValue)
    {
        setValue(newValue);
        return *this;
    }
};

/**
 * @brief Overload stream insertion operator (<<).
 * Writes only the clamped value to the stream. Bounds are ignored.
 * @param o The output stream reference.
 * @param s The ClampedScalar object.
 * @return Reference to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& o, const sofa::type::ClampedScalar<T>& s)
{
    return o << s.getValue();
}

/**
 * @brief Overload stream extraction operator (>>).
 * Reads a raw value from the stream and clamps it against the stored bounds before updating the object's state.
 * Bounds are not read or written during this operation.
 * @param i The input stream reference.
 * @param s The ClampedScalar object to be updated.
 * @return Reference to the input stream.
 */
template <typename T>
std::istream& operator>>(std::istream& i, sofa::type::ClampedScalar<T>& s)
{
    T rawValue{};
    i >> rawValue;
    s.setValue(rawValue);
    return i;
}

#if !defined(SOFA_TYPE_CLAMPEDSCALAR_CPP)
extern template class SOFA_TYPE_API ClampedScalar<double>;
extern template class SOFA_TYPE_API ClampedScalar<float>;
#endif

}


