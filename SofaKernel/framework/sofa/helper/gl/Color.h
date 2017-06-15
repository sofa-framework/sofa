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
#ifndef SOFA_HELPER_GL_COLOR_H
#define SOFA_HELPER_GL_COLOR_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/helper.h>

/// Forward declaration
namespace sofa {
    namespace helper {
        namespace types {
            class RGBAColor;
        }
    }
}


namespace sofa
{

namespace helper
{

namespace gl
{

class SOFA_HELPER_API Color
{
public:
    static void set(const sofa::helper::types::RGBAColor& color) ;

    static void setHSVA( float h, float s, float v, float a );
    static void getHSVA( float* rgba, float h, float s, float v, float a );

private:
    Color();
    ~Color();
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
