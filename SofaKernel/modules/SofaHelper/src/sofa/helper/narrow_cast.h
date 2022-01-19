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
#include <exception>
#include <utility>

namespace sofa::helper
{
    /// If true, narrow_cast will check if the value changed after the narrow conversion. Otherwise, no check is performed
#if defined(NDEBUG)
    constexpr bool forceNarrowCastChecking = false;
#else
    constexpr bool forceNarrowCastChecking = true;
#endif


    /**
     * \brief Explicit narrow conversion
     * Inspired by the Guidelines Support Library (https://github.com/microsoft/GSL)
     * \tparam T Target type
     * \tparam U Source type
     * \param u Value to cast
     * \return The value converted to the target type
     */
    template <class T, class U>
    constexpr T narrow_cast_nocheck(U&& u) noexcept
    {
        return static_cast<T>(std::forward<U>(u));
    }

    struct narrowing_error : public std::exception
    {
        const char* what() const noexcept override { return "narrowing_error"; }
    };

    /**
     * Explicit narrow conversion checking that the value is unchanged by the cast.
     * If the value changed, an exception is thrown
     * Inspired by the Guidelines Support Library (https://github.com/microsoft/GSL)
     */
    template <class T, class U>
    constexpr T narrow_cast_check(U u)
    {
        const T t = narrow_cast_nocheck<T>(u);

        using U_decay = std::decay_t<U>;

        if (static_cast<U_decay>(t) != u)
        {
            throw narrowing_error{};
        }

        if constexpr (std::is_arithmetic_v<T> && std::is_signed_v<T> != std::is_signed_v<U_decay>)
        {
            if ((t < T{}) != (u < U_decay{}))
            {
                throw narrowing_error{};
            }
        }

        return t;
    }

    /**
     * \brief Explicit narrow conversion
     * Inspired by the Guidelines Support Library (https://github.com/microsoft/GSL)
     * \tparam T Target type
     * \tparam U Source type
     * \param u Value to cast
     * \return The value converted to the target type
     */
    template <class T, class U>
    constexpr T narrow_cast(U&& u)
    {
        if constexpr (forceNarrowCastChecking)
        {
            return narrow_cast_check<T, U>(std::forward<U>(u));
        }
        else
        {
            return narrow_cast_nocheck<T, U>(std::forward<U>(u));
        }
    }


}
