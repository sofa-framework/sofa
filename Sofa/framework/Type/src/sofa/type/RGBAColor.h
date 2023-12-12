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

namespace sofa::type
{

#define RGBACOLOR_EQUALITY_THRESHOLD 1e-6

/**
 *  \brief encode a 4 RGBA component color
 */
class SOFA_TYPE_API RGBAColor
{
public:
    static constexpr sofa::Size NumberOfComponents = 4;
    using ComponentArray = std::array<float, NumberOfComponents>;

    constexpr RGBAColor()
        : m_components{ 1.f, 1.f, 1.f, 1.f } {}

    constexpr explicit RGBAColor(const std::array<float, NumberOfComponents>& c)
        : m_components(c) {}

    constexpr RGBAColor(float r, float g, float b, float a)
        : m_components{ r, g, b, a } {}

    // compat
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    RGBAColor(const type::fixed_array<float, NumberOfComponents>& c);
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    RGBAColor(const type::Vec4f& c);

    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const type::fixed_array<float, 4>& color);
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const type::fixed_array<double, 4>& color);
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const Vec4f& color);
    SOFA_ATTRIBUTE_DEPRECATED__RGBACOLOR_AS_FIXEDARRAY()
    static RGBAColor fromVec4(const Vec4d& color);

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
    using value_type = float;
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


constexpr RGBAColor g_white     {1.0f,1.0f,1.0f,1.0f};
constexpr RGBAColor g_black     {0.0f,0.0f,0.0f,1.0f};
constexpr RGBAColor g_red       {1.0f,0.0f,0.0f,1.0f};
constexpr RGBAColor g_green     {0.0f,1.0f,0.0f,1.0f};
constexpr RGBAColor g_blue      {0.0f,0.0f,1.0f,1.0f};
constexpr RGBAColor g_cyan      {0.0f,1.0f,1.0f,1.0f};
constexpr RGBAColor g_magenta   {1.0f,0.0f,1.0f,1.0f};
constexpr RGBAColor g_yellow    {1.0f,1.0f,0.0f,1.0f};
constexpr RGBAColor g_gray      {0.5f,0.5f,0.5f,1.0f};
constexpr RGBAColor g_darkgray  {0.25f,0.25f,0.25f,1.0f};
constexpr RGBAColor g_lightgray {0.75f,0.75f,0.75f,1.0f};

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

} // namespace sofa::type
