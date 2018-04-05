/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_QT_INFORMATIONONPICKCALLBACK
#define SOFA_GUI_QT_INFORMATIONONPICKCALLBACK

#include <sofa/gui/PickHandler.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/gui/ColourPickingVisitor.h>

namespace sofa
{
namespace component
{
namespace collision
{
struct BodyPicked;
}
}
namespace gui
{
namespace qt
{
namespace viewer
{
class SofaViewer;
}

class RealGUI;

class InformationOnPickCallBack: public CallBackPicker
{
public:
    InformationOnPickCallBack();
    InformationOnPickCallBack(RealGUI *g);
    void execute(const sofa::component::collision::BodyPicked &body);
protected:
    RealGUI *gui;
};


class ColourPickingRenderCallBack : public sofa::gui::CallBackRender
{
public:
    ColourPickingRenderCallBack();
    ColourPickingRenderCallBack(viewer::SofaViewer* viewer);
    void render(ColourPickingVisitor::ColourCode code);
protected:
    viewer::SofaViewer* _viewer;

};
}
}
}

#endif // SOFA_GUI_QT_INFORMATIONONPICKCALLBACK
