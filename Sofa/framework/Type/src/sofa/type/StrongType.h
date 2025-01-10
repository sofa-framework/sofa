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

namespace sofa::type
{

/**
 *  A template class to create strong, type-safe aliases for existing types,
 *  with optional extensible functionalities.
 *
 * The StrongType class is designed to add type safety by creating wrapper types
 * around primitive types or existing classes. By defining unique, named types
 * for different uses of the same underlying type, StrongType helps prevent
 * mistakes caused by passing incorrect parameters of the same type.
 *
 * For example, instead of passing two `int` parameters to a function (e.g., for
 * "width" and "height"), StrongType allows you to define distinct types for
 * width and height:
 *
 * @code
 * using Width = StrongType<int, struct WidthTag>;
 * using Height = StrongType<int, struct HeightTag>;
 *
 * void resize(Width width, Height height);
 *
 * resize(Width{5}, Height{10});  // Correct
 * resize(Height{10}, Width{5});  // Compilation error
 * @endcode
 *
 * Strongly inspired by https://github.com/joboccara/NamedType
 *
 * @tparam TUnderlyingType The type to be wrapped by StrongType
 * @tparam UniqueIdentifyingTag A unique tag type to create a distinct StrongType from other wrappers around the same `TUnderlyingType`.
 * @tparam Functionalities A set of optional functionalities to extend StrongType
 */
template<class TUnderlyingType, class UniqueIdentifyingTag, template <typename> class... Functionalities>
struct StrongType : Functionalities<StrongType<TUnderlyingType, UniqueIdentifyingTag, Functionalities...>>...
{
    using UnderlyingType = TUnderlyingType;

    constexpr explicit StrongType(const UnderlyingType& v) : m_value(v) {}

    [[nodiscard]] constexpr UnderlyingType& get() noexcept
    {
        return m_value;
    }

    [[nodiscard]] constexpr const std::remove_reference_t<UnderlyingType>& get() const
    {
        return m_value;
    }

private:
    UnderlyingType m_value;
};

namespace functionality
{

template <typename T, template <typename> class Functionality>
struct UnderlyingAccess
{
    [[nodiscard]] T& underlying()
    {
        return static_cast<T&>(*this);
    }
    [[nodiscard]] constexpr T const& underlying() const
    {
        return static_cast<T const&>(*this);
    }
};

template <typename T>
struct PreIncrementable : UnderlyingAccess<T, PreIncrementable>
{
    constexpr T& operator++()
    {
        ++static_cast<T&>(*this).get();
        return this->underlying();
    }
};

template <typename T>
struct PostIncrementable : UnderlyingAccess<T, PostIncrementable>
{
    constexpr T operator++(int)
    {
        return T(this->underlying().get()++);
    }
};


template <typename T>
struct Incrementable
    : PreIncrementable<T>
    , PostIncrementable<T>
{
    using PostIncrementable<T>::operator++;
    using PreIncrementable<T>::operator++;
};

template <typename T>
struct PreDecrementable : UnderlyingAccess<T, PreDecrementable>
{
    constexpr T& operator--()
    {
        --this->underlying().get();
        return this->underlying();
    }
};

template <typename T>
struct PostDecrementable : UnderlyingAccess<T, PostDecrementable>
{
    constexpr T operator--(int)
    {
        return T( this->underlying().get()-- );
    }
};


template <typename T>
struct Decrementable
    : PreDecrementable<T>
    , PostDecrementable<T>
{
    using PostDecrementable<T>::operator--;
    using PreDecrementable<T>::operator--;
};

template <typename T>
struct BinaryAddable : UnderlyingAccess<T, BinaryAddable>
{
    constexpr T operator+(T const& other) const
    {
        return T(this->underlying().get() + other.get());
    }
    T& operator+=(T const& other)
    {
        this->underlying().get() += other.get();
        return this->underlying();
    }
};

template <typename T>
struct UnaryAddable : UnderlyingAccess<T, UnaryAddable>
{
    constexpr T operator+() const
    {
        return T(+this->underlying().get());
    }
};

template <typename T>
struct Addable
    : BinaryAddable<T>
    , UnaryAddable<T>
{
    using BinaryAddable<T>::operator+;
    using UnaryAddable<T>::operator+;
};

template <typename T>
struct BinarySubtractable : UnderlyingAccess<T, BinarySubtractable>
{
    [[nodiscard]] constexpr T operator-(T const& other) const
    {
        return T(this->underlying().get() - other.get());
    }
    constexpr T& operator-=(T const& other)
    {
        this->underlying().get() -= other.get();
        return this->underlying();
    }
};

template <typename T>
struct UnarySubtractable : UnderlyingAccess<T, UnarySubtractable>
{
    [[nodiscard]] constexpr T operator-() const
    {
        return T(-this->underlying().get());
    }
};

template <typename T>
struct Subtractable
    : BinarySubtractable<T>
    , UnarySubtractable<T>
{
    using UnarySubtractable<T>::operator-;
    using BinarySubtractable<T>::operator-;
};


template <typename T>
struct Multiplicable : UnderlyingAccess<T, Multiplicable>
{
    [[nodiscard]] constexpr T operator*(T const& other) const
    {
        return T(this->underlying().get() * other.get());
    }
    constexpr T& operator*=(T const& other)
    {
        this->underlying().get() *= other.get();
        return this->underlying();
    }
};

template <typename T>
struct Divisible : UnderlyingAccess<T, Divisible>
{
    [[nodiscard]] constexpr T operator/(T const& other) const
    {
        return T(this->underlying().get() / other.get());
    }
    constexpr T& operator/=(T const& other)
    {
        this->underlying().get() /= other.get();
        return this->underlying();
    }
};

template <typename T>
struct Modulable : UnderlyingAccess<T, Modulable>
{
    [[nodiscard]] constexpr T operator%(T const& other) const
    {
        return T(this->underlying().get() % other.get());
    }
    constexpr T& operator%=(T const& other)
    {
        this->underlying().get() %= other.get();
        return this->underlying();
    }
};

template <typename T>
struct Comparable : UnderlyingAccess<T, Comparable>
{
    [[nodiscard]] constexpr bool operator<(Comparable<T> const& other) const
    {
        return this->underlying().get() < other.underlying().get();
    }
    [[nodiscard]] constexpr bool operator>(Comparable<T> const& other) const
    {
        return other.underlying().get() < this->underlying().get();
    }
    [[nodiscard]] constexpr bool operator<=(Comparable<T> const& other) const
    {
        return !(other < *this);
    }
    [[nodiscard]] constexpr bool operator>=(Comparable<T> const& other) const
    {
        return !(*this < other);
    }
    [[nodiscard]] constexpr bool operator==(Comparable<T> const& other) const
    {
        return !(*this < other) && !(other < *this);
    }
    [[nodiscard]] constexpr bool operator!=(Comparable<T> const& other) const
    {
        return !(*this == other);
    }
};

template <typename T>
struct Arithmetic
    : Incrementable<T>
    , Decrementable<T>
    , Addable<T>
    , Subtractable<T>
    , Multiplicable<T>
    , Divisible<T>
    , Modulable<T>
    , Comparable<T>
{
};

}

}
