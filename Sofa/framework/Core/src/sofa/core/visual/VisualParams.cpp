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
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/helper/BackTrace.h>
#include <cassert>
#include <iostream>

namespace sofa::core::visual
{

VisualParams::VisualParams()
    : m_viewport(sofa::type::make_array(0,0,0,0))
    , m_zNear(0)
    , m_zFar(0)
    , m_cameraType(PERSPECTIVE_TYPE)
    , m_pass(Std)
    , m_drawTool(nullptr)
    //, m_boundFrameBuffer(nullptr)
    , m_x (vec_id::read_access::position)
    , m_v (vec_id::read_access::velocity)
    , m_supportedAPIs(0)
{
    m_displayFlags.setShowVisualModels(true); // BUGFIX: visual models are visible by default
}

/// Get the default VisualParams, to be used to provide a default values for method parameters
VisualParams* VisualParams::defaultInstance()
{
    thread_local VisualParams threadParams;
    return &threadParams;
}
} // namespace sofa::core::visual
