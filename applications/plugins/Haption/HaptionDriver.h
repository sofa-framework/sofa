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
#ifndef SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H
#define SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H

//Haption include
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/gl/component/rendering3d/OglModel.h>
#include <sofa/component/mapping/nonlinear/RigidMapping.h>
#include <sofa/component/controller/Controller.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/haptics/MechanicalStateForceFeedback.h>
#include <sofa/component/haptics/NullForceFeedbackT.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <cstring>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <math.h>
#include "virtuoseAPI.h"


namespace sofa
{
namespace simulation { class Node; }

namespace component
{
namespace visualModel { class OglModel; }

namespace controller
{

class ForceFeedback;

using namespace sofa::defaulttype;
using core::objectmodel::Data;

typedef struct
{
    simulation::Node *node;
    sofa::gl::component::rendering3d::OglModel *visu;
    sofa::component::mapping::nonlinear::RigidMapping< Rigid3dTypes , Vec3fTypes  > *mapping;
} VisualComponent;

typedef struct
{
    VirtContext m_virtContext;
    haptics::MechanicalStateForceFeedback<Rigid3dTypes>* forceFeedback;
    float scale;
    float torqueScale;
    float forceScale;
} HaptionData;


class HaptionDriver : public Controller
{

public:
    SOFA_CLASS(HaptionDriver, Controller);
    typedef RigidTypes::VecCoord VecCoord;

    Data<double> scale; ///< Default scale applied to the Haption Coordinates. 
    Data<bool> state_button; ///< state of the first button
    Data<bool> haptionVisu; ///< Visualize the position of the interface in the virtual scene
    Data<VecCoord> posBase; ///< Position of the interface base in the scene world coordinates
    Data<double> torqueScale; ///< Default scale applied to the Haption torque. 
    Data<double> forceScale; ///< Default scale applied to the Haption force. 
    Data< std::string > ip_haption; ///< ip of the device

    HaptionDriver();
    virtual ~HaptionDriver();

    void init() override;
    void reinit() override;
    void bwdInit() override;
    void reset() override;
    void handleEvent(core::objectmodel::Event *) override;
    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *) override;
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *) override;
    void onAnimateBeginEvent();
    int initDevice(char* ip);
    void closeDevice();
    static void haptic_callback(VirtContext, void *);

    void setForceFeedback(haptics::MechanicalStateForceFeedback<Rigid3dTypes>* ff);

private:

    HaptionData myData;
    VirtIndexingType m_indexingMode;
    VirtCommandType m_typeCommand;
    float m_speedFactor;
    float m_forceFactor;
    float haptic_time_step;
    int connection_device;
    sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *rigidDOF;
    bool initCallback;
    simulation::Node *nodeHaptionVisual;
    sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *visualHaptionDOF;
    simulation::Node *nodeAxesVisual;
    sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *visualAxesDOF;
    VisualComponent visualNode[5];

    float oldScale;
    bool changeScale;
    bool visuAxes;
    bool modX,modY,modZ,modS;
    bool visuActif;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H
