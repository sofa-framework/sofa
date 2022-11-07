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
#include <sofa/gui/qt/config.h>
#include <sofa/gui/common/OperationFactory.h>

#include <sofa/gui/common/PickHandler.h>

#include <sofa/gui/common/ColourPickingVisitor.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/gl/FrameBufferObject.h>

namespace sofa::component::collision::model
{
    class RayCollisionModel;
} // namespace sofa::component::collision::model

namespace sofa::gui::qt
{

class SOFA_GUI_QT_API GLPickHandler : public common::PickHandler
{
    typedef PickHandler Inherit;
    typedef sofa::component::collision::model::RayCollisionModel MouseCollisionModel;
    typedef sofa::component::statecontainer::MechanicalObject< defaulttype::Vec3Types > MouseContainer;

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

    common::BodyPicked findCollisionUsingColourCoding(const type::Vec3& origin, const type::Vec3& direction) override;

protected:
    bool _fboAllocated;
    gl::FrameBufferObject _fbo;
    gl::fboParameters     _fboParams;

};

} // namespace sofa::gui::qt
