/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Interface: OglRenderingSRGB
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_OGLRENDERING_SRGB_H
#define SOFA_OGLRENDERING_SRGB_H

#include <sofa/SofaGeneral.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief The utility to enable/disable sRGB rendering
 */

class SOFA_OPENGL_VISUAL_API OglRenderingSRGB : public core::visual::VisualManager
{
public:
    SOFA_CLASS(OglRenderingSRGB, core::visual::VisualManager);

    void fwdDraw(core::visual::VisualParams* );
    void bwdDraw(core::visual::VisualParams* );
};

}//namespace visualmodel

}//namespace component

}//namespace sofa

#endif //SOFA_RENDERING_SRGB_H
