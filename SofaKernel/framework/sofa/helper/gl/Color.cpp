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
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/helper/logging/Messaging.h>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace gl
{
using sofa::helper::types::RGBAColor ;

void Color::setHSVA( float h, float s, float v, float a )
{
    msg_deprecated("gl::Color") << "The setHSVA function is deprecated. "
                               "Using deprecated function may result in incorrect behavior as well as loss of performance"
                               "To remove this error message you can update the setHSVA function with the following: "
                               "Color::set( RGBAColor::fromHSVA(h,s,v,a) ); " ;

   glColor4fv( RGBAColor::fromHSVA(h,s,v,a).data() );
}

void Color::getHSVA( float* rgba, float h, float s, float v, float a )
{
    assert(rgba!=nullptr) ;

    msg_deprecated("gl::Color") << "The getHSVA function is deprecated. "
                               "Using deprecated function may result in incorrect behavior as well as loss of performance"
                               "To remove this error message you can update the getHSVA function with the following: "
                               "RGBAColor::fromHSVA(h,s,v,a).data() " ;
    RGBAColor tmp=RGBAColor::fromHSVA(h,s,v,a);
    for(unsigned int i=0; i<4;i++){
        rgba[i] = tmp[i] ;
    }
}


} // namespace gl

} // namespace helper

} // namespace sofa

