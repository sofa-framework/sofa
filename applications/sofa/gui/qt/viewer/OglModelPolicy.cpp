/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/gui/qt/viewer/OglModelPolicy.h>
#include <sofa/core/visual/DrawToolGL.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

void OglModelPolicy::load()
{
    drawTool = std::unique_ptr<sofa::core::visual::DrawTool>(new sofa::core::visual::DrawToolGL());

    // Replace generic visual models with OglModel
    sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true,
            &classVisualModel);
    vparams->drawTool() = drawTool.get();
    vparams->setSupported(sofa::core::visual::API_OpenGL);
}

void OglModelPolicy::unload()
{
    sofa::core::ObjectFactory::ResetAlias("VisualModel", classVisualModel);
    vparams->drawTool() = nullptr;

}

} // namespace viewer
} // namespace qt
} // namespace gui
} // namespace sofa

