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
#ifndef SOFA_GUI_GLPICKHANDLER_H
#define SOFA_GUI_GLPICKHANDLER_H

#include <sofa/gui/qt/SofaGuiQt.h>
#include <sofa/gui/OperationFactory.h>

#include <sofa/gui/PickHandler.h>

#include <sofa/gui/ColourPickingVisitor.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/gl/FrameBufferObject.h>

namespace sofa
{
namespace component
{
namespace collision
{
    class ComponentMouseInteraction;
    class RayCollisionModel;
}
namespace configurationsetting
{
    class MouseButtonSetting;
}
}


namespace gui
{

class SOFA_SOFAGUIQT_API GLPickHandler : public PickHandler
{
    typedef PickHandler Inherit;
    typedef sofa::component::collision::RayCollisionModel MouseCollisionModel;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > MouseContainer;

public:
    enum PickingMethod
    {
        RAY_CASTING,
        SELECTION_BUFFER
    };

    GLPickHandler(double defaultLength = 1000000);
    virtual ~GLPickHandler() override;

    void allocateSelectionBuffer(int width, int height) override;
    void destroySelectionBuffer() override;

    BodyPicked findCollisionUsingColourCoding(const defaulttype::Vector3& origin, const defaulttype::Vector3& direction) override;

protected:
    bool _fboAllocated;
    sofa::helper::gl::FrameBufferObject _fbo;
    sofa::helper::gl::fboParameters     _fboParams;

};
}
}

#endif
