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
#pragma once
#include <sofa/gui/common/PickHandler.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/gui/common/ColourPickingVisitor.h>

namespace sofa::component::collision
{
    struct BodyPicked;
} // namespace sofa::component::collision

namespace sofa::gui::qt::viewer
{
    class SofaViewer;
} // namespace sofa::gui::qt::viewer

namespace sofa::gui::qt
{
class RealGUI;

class InformationOnPickCallBack: public common::CallBackPicker
{
public:
    using BodyPicked = sofa::gui::component::performer::BodyPicked;
    InformationOnPickCallBack();
    InformationOnPickCallBack(RealGUI *g);
    void execute(const BodyPicked &body) override;
protected:
    RealGUI *gui;
};


class ColourPickingRenderCallBack : public sofa::gui::common::CallBackRender
{
public:
    ColourPickingRenderCallBack();
    ColourPickingRenderCallBack(viewer::SofaViewer* viewer);
    void render(common::ColourPickingVisitor::ColourCode code) override;
protected:
    viewer::SofaViewer* _viewer;

};

} // namespace sofa::gui::qt
