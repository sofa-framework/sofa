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
#ifndef SOFA_HELPER_GL_RAII_H
#define SOFA_HELPER_GL_RAII_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/system/gl.h>
#include <sofa/helper/helper.h>

/* Opengl Resource Acquisition Is Initialisation */
/* with this tool, we know at any moment what is the state of the openGL machine */

namespace sofa
{

namespace helper
{

namespace gl
{

template <GLenum Flag>
struct Enable
{
    GLboolean state;

    Enable ()
    {
        state = glIsEnabled(Flag);
        // state contains the state of the Flag
        // if this flag is activated, we haven't to reactivate it
        if (!state)
            glEnable (Flag);
    };

    ~Enable ()
    {
        if (!state)
            glDisable (Flag);
    };
};

template <GLenum Flag>
struct Disable
{

    GLboolean state;

    Disable ()
    {
        state = glIsEnabled(Flag);
        // state contains the state of the Flag
        // if this flag is activated, we haven't to reactivate it
        if (state)
            glDisable (Flag);
    };

    ~Disable ()
    {
        if (state)
            glEnable (Flag);
    };
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
