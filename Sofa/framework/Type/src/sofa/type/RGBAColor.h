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

#include <sofa/type/fixed_array.h>

#include <ostream>
#include <istream>
#include <string>
#include <cmath>


namespace sofa::type
{

using sofa::type::fixed_array ;


#define RGBACOLOR_EQUALITY_THRESHOLD 1e-6

/**
 *  \brief encode a 4 RGBA component color
 */
class SOFA_TYPE_API RGBAColor : public fixed_array<float, 4>
{
public:
    static RGBAColor fromString(const std::string& str);
    static RGBAColor fromFloat(float r, float g, float b, float a);
    static RGBAColor fromVec4(const fixed_array<float, 4>& color);
    static RGBAColor fromVec4(const fixed_array<double, 4>& color);

    static RGBAColor fromHSVA(float h, float s, float v, float a);

    static bool read(const std::string& str, RGBAColor& color);

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

    constexpr float& r()
    {
        return this->elems[0];
    }

    constexpr float& g()
    {
        return this->elems[1];
    }

    constexpr float& b()
    {
        return this->elems[2];
    }

    constexpr float& a()
    {
        return this->elems[3];
    }

    constexpr const float& r() const
    {
        return this->elems[0];
    }

    constexpr const float& g() const
    {
        return this->elems[1];
    }

    constexpr const float& b() const
    {
        return this->elems[2];
    }

    constexpr const float& a() const
    {
        return this->elems[3];
    }

    constexpr void r(const float r)
    {
        this->elems[0] = r;
    }

    constexpr void g(const float g)
    {
        this->elems[1] = g;
    }

    constexpr void b(const float b)
    {
        this->elems[2] = b;
    }

    constexpr void a(const float a)
    {
        this->elems[3] = a;
    }

    void set(float r, float g, float b, float a);

    bool operator==(const fixed_array<float, 4>& b) const
    {
        for (int i = 0; i < 4; i++)
            if (fabs(this->elems[i] - b[i]) > RGBACOLOR_EQUALITY_THRESHOLD)
                return false;
        return true;
    }

    bool operator!=(const fixed_array<float, 4>& b) const
    {
        for (int i = 0; i < 4; i++)
            if (fabs(this->elems[i] - b[i]) > RGBACOLOR_EQUALITY_THRESHOLD)
                return true;
        return false;
    }

    RGBAColor operator*(float f) const;

    friend SOFA_TYPE_API std::ostream& operator<<(std::ostream& i, const RGBAColor& t);
    friend SOFA_TYPE_API std::istream& operator>>(std::istream& i, RGBAColor& t);

    constexpr RGBAColor()
        : fixed_array<float, 4>(1.f, 1.f, 1.f, 1.f)
    {}

    constexpr RGBAColor(const fixed_array<float, 4>& c)
        : fixed_array<float, 4>(c)
    {}

    constexpr RGBAColor(float r, float g, float b, float a)
        : fixed_array<float, 4>(r, g, b, a)
    {}

};


inline constexpr RGBAColor g_white{1.0f, 1.0f, 1.0f, 1.0f};
inline constexpr RGBAColor g_black{0.0f, 0.0f, 0.0f, 1.0f};
inline constexpr RGBAColor g_red{1.0f, 0.0f, 0.0f, 1.0f};
inline constexpr RGBAColor g_green{0.0f, 1.0f, 0.0f, 1.0f};
inline constexpr RGBAColor g_blue{0.0f, 0.0f, 1.0f, 1.0f};
inline constexpr RGBAColor g_cyan{0.0f, 1.0f, 1.0f, 1.0f};
inline constexpr RGBAColor g_magenta{1.0f, 0.0f, 1.0f, 1.0f};
inline constexpr RGBAColor g_yellow{1.0f, 1.0f, 0.0f, 1.0f};
inline constexpr RGBAColor g_gray{0.5f, 0.5f, 0.5f, 1.0f};
inline constexpr RGBAColor g_darkgray{0.25f, 0.25f, 0.25f, 1.0f};
inline constexpr RGBAColor g_lightgray{0.75f, 0.75f, 0.75f, 1.0f};

constexpr const RGBAColor& RGBAColor::white()
{
    return g_white;
}

constexpr const RGBAColor& RGBAColor::black()
{
    return g_black;
}

constexpr const RGBAColor& RGBAColor::red()
{
    return g_red;
}

constexpr const RGBAColor& RGBAColor::green()
{
    return g_green;
}

constexpr const RGBAColor& RGBAColor::blue()
{
    return g_blue;
}

constexpr const RGBAColor& RGBAColor::cyan()
{
    return g_cyan;
}

constexpr const RGBAColor& RGBAColor::magenta()
{
    return g_magenta;
}

constexpr const RGBAColor& RGBAColor::yellow()
{
    return g_yellow;
}

constexpr const RGBAColor& RGBAColor::gray()
{
    return g_gray;
}

constexpr const RGBAColor& RGBAColor::darkgray()
{
    return g_darkgray;
}

constexpr const RGBAColor& RGBAColor::lightgray()
{
    return g_lightgray;
}

} // namespace sofa::type
