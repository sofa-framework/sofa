/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_RGBAHELPER_COLOR_H
#define SOFA_RGBAHELPER_COLOR_H
#include <iostream>

#include <string>

#include <sofa/helper/helper.h>
#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace helper
{

namespace types
{

#define RGBACOLOR_EQUALITY_THRESHOLD 1e-6

/**
 *  \brief encode a 4 RGBA component color
 */
class SOFA_HELPER_API RGBAColor : public fixed_array<float, 4>
{
public:
    static RGBAColor fromString(const std::string& str) ;
    static RGBAColor fromFloat(const float r, const float g, const float b, const float a) ;
    static RGBAColor fromVec4(const fixed_array<float, 4>& color) ;
    static RGBAColor fromVec4(const fixed_array<double, 4>& color) ;

    static RGBAColor fromHSVA(float h, float s, float v, float a) ;

    static bool read(const std::string& str, RGBAColor& color) ;

    static RGBAColor white()  { return RGBAColor(1.0,1.0,1.0,1.0); }
    static RGBAColor black()  { return RGBAColor(0.0,0.0,0.0,1.0); }
    static RGBAColor red()    { return RGBAColor(1.0,0.0,0.0,1.0); }
    static RGBAColor green()  { return RGBAColor(0.0,1.0,0.0,1.0); }
    static RGBAColor blue()   { return RGBAColor(0.0,0.0,1.0,1.0); }
    static RGBAColor cyan()   { return RGBAColor(0.0,1.0,1.0,1.0); }
    static RGBAColor magenta() { return RGBAColor(1.0,0.0,1.0,1.0); }
    static RGBAColor yellow()  { return RGBAColor(1.0,1.0,0.0,1.0); }
    static RGBAColor gray()    { return RGBAColor(0.5,0.5,0.5,1.0); }

    inline float& r(){ return this->elems[0] ; }
    inline float& g(){ return this->elems[1] ; }
    inline float& b(){ return this->elems[2] ; }
    inline float& a(){ return this->elems[3] ; }
    inline const float& r() const { return this->elems[0] ; }
    inline const float& g() const { return this->elems[1] ; }
    inline const float& b() const { return this->elems[2] ; }
    inline const float& a() const { return this->elems[3] ; }

    inline void r(const float r){ this->elems[0]=r; }
    inline void g(const float g){ this->elems[1]=g; }
    inline void b(const float b){ this->elems[2]=b; }
    inline void a(const float a){ this->elems[3]=a; }

    void set(float r, float g, float b, float a) ;

    bool operator==(const fixed_array<float,4>& b) const
    {
        for (int i=0; i<4; i++)
            if ( fabs( this->elems[i] - b[i] ) > RGBACOLOR_EQUALITY_THRESHOLD ) return false;
        return true;
    }

    bool operator!=(const fixed_array<float,4>& b) const
    {
        for (int i=0; i<4; i++)
            if ( fabs( this->elems[i] - b[i] ) > RGBACOLOR_EQUALITY_THRESHOLD ) return true;
        return false;
    }

    friend SOFA_HELPER_API std::ostream& operator<<(std::ostream& i, const RGBAColor& t) ;
    friend SOFA_HELPER_API std::istream& operator>>(std::istream& i, RGBAColor& t) ;

public:
    RGBAColor() ;
    RGBAColor(const fixed_array<float, 4>&) ;
    RGBAColor(const float r, const float g, const float b, const float a) ;

};

} // namespace types

} // namespace helper

} // namespace sofa


#endif

