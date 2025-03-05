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

#include <sofa/type/fwd.h>

#include <ostream>
#include <istream>
#include <string>
#include <cmath>
#include <array>
#include <algorithm>
#include <cassert>

#include <sofa/type/fixed_array_algorithms.h>
#include <sofa/type/StrongType.h>


namespace sofa::type
{

#define RGBACOLOR_EQUALITY_THRESHOLD 1e-6

/**
 *  \brief encode a 4 RGBA component color
 */
class SOFA_TYPE_API RGBAColor
{
public:
    using value_type = float;
    static constexpr sofa::Size NumberOfComponents = 4;
    using ComponentArray = std::array<value_type, NumberOfComponents>;

    constexpr RGBAColor()
        : m_components{ 1.f, 1.f, 1.f, 1.f } {}

    constexpr explicit RGBAColor(const std::array<float, NumberOfComponents>& c)
        : m_components(c) {}

    constexpr RGBAColor(float r, float g, float b, float a)
        : m_components{ r, g, b, a } {}

    // compat
    SOFA_ATTRIBUTE_DISABLED__RGBACOLOR_AS_FIXEDARRAY()
    RGBAColor(const type::fixed_array<float, NumberOfComponents>& c) = delete;
    SOFA_ATTRIBUTE_DISABLED__RGBACOLOR_AS_FIXEDARRAY()
    RGBAColor(const type::Vec4f& c) = delete;

    SOFA_ATTRIBUTE_DISABLED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const type::fixed_array<float, 4>& color) = delete;
    SOFA_ATTRIBUTE_DISABLED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const type::fixed_array<double, 4>& color) = delete;
    SOFA_ATTRIBUTE_DISABLED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const Vec4f& color) = delete;
    SOFA_ATTRIBUTE_DISABLED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const Vec4d& color) = delete;

    static RGBAColor fromString(const std::string& str);
    static RGBAColor fromFloat(float r, float g, float b, float a);
    static RGBAColor fromStdArray(const std::array<float, 4>& color);
    static RGBAColor fromStdArray(const std::array<double, 4>& color);
    static RGBAColor fromHSVA(float h, float s, float v, float a);

    static bool read(const std::string& str, RGBAColor& color) ;

    constexpr static const RGBAColor& white();
    constexpr static const RGBAColor& black();
    constexpr static const RGBAColor& red();
    constexpr static const RGBAColor& green();
    constexpr static const RGBAColor& blue();
    constexpr static const RGBAColor& cyan();
    constexpr static const RGBAColor& magenta();
    constexpr static const RGBAColor& yellow();
    constexpr static const RGBAColor& gray();
    constexpr static const RGBAColor& darkgray();
    constexpr static const RGBAColor& lightgray();
    constexpr static const RGBAColor& orange();
    constexpr static const RGBAColor& purple();
    constexpr static const RGBAColor& pink();
    constexpr static const RGBAColor& brown();
    constexpr static const RGBAColor& lime();
    constexpr static const RGBAColor& teal();
    constexpr static const RGBAColor& navy();
    constexpr static const RGBAColor& olive();
    constexpr static const RGBAColor& maroon();
    constexpr static const RGBAColor& silver();
    constexpr static const RGBAColor& gold();

    /// @brief enlight a color by a given factor.
    static RGBAColor lighten(const RGBAColor& in, const SReal factor);

    constexpr float& r(){ return this->m_components[0] ; }
    constexpr float& g(){ return this->m_components[1] ; }
    constexpr float& b(){ return this->m_components[2] ; }
    constexpr float& a(){ return this->m_components[3] ; }
    constexpr const float& r() const { return this->m_components[0] ; }
    constexpr const float& g() const { return this->m_components[1] ; }
    constexpr const float& b() const { return this->m_components[2] ; }
    constexpr const float& a() const { return this->m_components[3] ; }

    constexpr void r(const float r){ this->m_components[0]=r; }
    constexpr void g(const float g){ this->m_components[1]=g; }
    constexpr void b(const float b){ this->m_components[2]=b; }
    constexpr void a(const float a){ this->m_components[3]=a; }

    // operator[]
    constexpr float& operator[](std::size_t i)
    {
        assert(i < NumberOfComponents && "index in RGBAColor must be smaller than 4");
        return m_components[i];
    }
    constexpr const float& operator[](std::size_t i) const
    {
        assert(i < NumberOfComponents && "index in RGBAColor must be smaller than 4");
        return m_components[i];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr float& get() & noexcept requires (I < 4)
    {
        return m_components[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const float& get() const& noexcept requires (I < 4)
    {
        return m_components[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr float&& get() && noexcept requires (I < 4)
    {
        return std::move(m_components[I]);
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const float&& get() const&& noexcept requires (I < 4)
    {
        return std::move(m_components[I]);
    }

    void set(float r, float g, float b, float a) ;

    bool operator==(const RGBAColor& b) const
    {
        for (int i=0; i<4; i++)
            if ( fabs( this->m_components[i] - b[i] ) > RGBACOLOR_EQUALITY_THRESHOLD ) return false;
        return true;
    }

    bool operator!=(const RGBAColor& b) const
    {
        for (int i=0; i<4; i++)
            if ( fabs( this->m_components[i] - b[i] ) > RGBACOLOR_EQUALITY_THRESHOLD ) return true;
        return false;
    }

    bool operator<(const RGBAColor& b) const
    {
        for (int i = 0; i < 4; i++)
            if (this->m_components[i] < b[i]) return true;
        return false;
    }

    RGBAColor operator*(float f) const;

    friend SOFA_TYPE_API std::ostream& operator<<(std::ostream& i, const RGBAColor& t) ;
    friend SOFA_TYPE_API std::istream& operator>>(std::istream& i, RGBAColor& t) ;

    // direct access to data
    constexpr const float* data() const noexcept
    {
        return m_components.data();
    }

    /// direct access to array
    constexpr const ComponentArray& array() const noexcept
    {
        return m_components;
    }

    /// direct access to array
    constexpr ComponentArray& array() noexcept
    {
        return m_components;
    }

    static constexpr RGBAColor clamp(const RGBAColor& color, float min, float max)
    {
        RGBAColor result{};

        for(sofa::Size i = 0 ; i < NumberOfComponents; ++i)
        {
            result[i] = std::clamp(color[i], min, max);
        }
        return result;
    }

    constexpr ComponentArray::iterator begin() noexcept
    {
        return m_components.begin();
    }
    constexpr ComponentArray::const_iterator begin() const noexcept
    {
        return m_components.begin();
    }

    constexpr ComponentArray::iterator end() noexcept
    {
        return m_components.end();
    }
    constexpr ComponentArray::const_iterator end() const noexcept
    {
        return m_components.end();
    }

    static constexpr sofa::Size static_size = NumberOfComponents;
    static constexpr sofa::Size size() { return static_size; }
    using size_type = sofa::Size;

private:
    ComponentArray m_components;
};

constexpr RGBAColor operator-(const RGBAColor& l, const RGBAColor& r)
{
    return sofa::type::pairwise::operator-(l, r);
}

constexpr RGBAColor operator+(const RGBAColor& l, const RGBAColor& r)
{
    return sofa::type::pairwise::operator+(l, r);
}

constexpr RGBAColor operator/(const RGBAColor& l, const float div)
{
    RGBAColor result{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        result[i] = l[i] / div;
    }
    return result;
}


constexpr RGBAColor g_white     {1.0f, 1.0f, 1.0f, 1.0f};
constexpr RGBAColor g_black     {0.0f, 0.0f, 0.0f, 1.0f};
constexpr RGBAColor g_red       {1.0f, 0.0f, 0.0f, 1.0f};
constexpr RGBAColor g_green     {0.0f, 1.0f, 0.0f, 1.0f};
constexpr RGBAColor g_blue      {0.0f, 0.0f, 1.0f, 1.0f};
constexpr RGBAColor g_cyan      {0.0f, 1.0f, 1.0f, 1.0f};
constexpr RGBAColor g_magenta   {1.0f, 0.0f, 1.0f, 1.0f};
constexpr RGBAColor g_yellow    {1.0f, 1.0f, 0.0f, 1.0f};
constexpr RGBAColor g_gray      {0.5f, 0.5f, 0.5f, 1.0f};
constexpr RGBAColor g_darkgray  {0.25f, 0.25f,0.25f, 1.0f};
constexpr RGBAColor g_lightgray {0.75f, 0.75f,0.75f, 1.0f};
constexpr RGBAColor g_orange    { 1.0f, 0.5f, 0.0f, 1.0f };
constexpr RGBAColor g_purple    { 0.5f, 0.0f, 0.5f, 1.0f };
constexpr RGBAColor g_pink      { 1.0f, 0.0f, 0.5f, 1.0f };
constexpr RGBAColor g_brown     { 0.6f, 0.3f, 0.0f, 1.0f };
constexpr RGBAColor g_lime      { 0.5f, 1.0f, 0.0f, 1.0f };
constexpr RGBAColor g_teal      { 0.0f, 0.5f, 0.5f, 1.0f };
constexpr RGBAColor g_navy      { 0.0f, 0.0f, 0.5f, 1.0f };
constexpr RGBAColor g_olive     { 0.5f, 0.5f, 0.0f, 1.0f };
constexpr RGBAColor g_maroon    { 0.5f, 0.0f, 0.0f, 1.0f };
constexpr RGBAColor g_silver    { 0.75f, 0.75f, 0.75f, 1.0f };
constexpr RGBAColor g_gold      { 1.0f, 0.84f, 0.0f, 1.0f };

constexpr const RGBAColor& RGBAColor::white()    { return g_white;     }
constexpr const RGBAColor& RGBAColor::black()    { return g_black;     }
constexpr const RGBAColor& RGBAColor::red()      { return g_red;       }
constexpr const RGBAColor& RGBAColor::green()    { return g_green;     }
constexpr const RGBAColor& RGBAColor::blue()     { return g_blue;      }
constexpr const RGBAColor& RGBAColor::cyan()     { return g_cyan;      }
constexpr const RGBAColor& RGBAColor::magenta()  { return g_magenta;   }
constexpr const RGBAColor& RGBAColor::yellow()   { return g_yellow;    }
constexpr const RGBAColor& RGBAColor::gray()     { return g_gray;      }
constexpr const RGBAColor& RGBAColor::darkgray() { return g_darkgray;  }
constexpr const RGBAColor& RGBAColor::lightgray(){ return g_lightgray; }
constexpr const RGBAColor& RGBAColor::orange()   { return g_orange; }
constexpr const RGBAColor& RGBAColor::purple()   { return g_purple; }
constexpr const RGBAColor& RGBAColor::pink()     { return g_pink  ; }
constexpr const RGBAColor& RGBAColor::brown()    { return g_brown ; }
constexpr const RGBAColor& RGBAColor::lime()     { return g_lime  ; }
constexpr const RGBAColor& RGBAColor::teal()     { return g_teal  ; }
constexpr const RGBAColor& RGBAColor::navy()     { return g_navy  ; }
constexpr const RGBAColor& RGBAColor::olive()    { return g_olive ; }
constexpr const RGBAColor& RGBAColor::maroon()   { return g_maroon; }
constexpr const RGBAColor& RGBAColor::silver()   { return g_silver; }
constexpr const RGBAColor& RGBAColor::gold()     { return g_gold  ; }


} // namespace sofa::type


namespace std
{

template<>
struct tuple_size<::sofa::type::RGBAColor > : integral_constant<size_t, ::sofa::type::RGBAColor::NumberOfComponents> {};

template<std::size_t I>
struct tuple_element<I, ::sofa::type::RGBAColor >
{
    using type = ::sofa::type::RGBAColor::value_type;
};

}
