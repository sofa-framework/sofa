/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>
#include <sofa/core/behavior/BaseController.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaUserInteraction/Controller.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaHaptics/MechanicalStateForceFeedback.h>
#include <SofaHaptics/NullForceFeedbackT.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <cstring>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
//#include <sofa/core/objectmodel/MouseEvent.h>
#include <math.h>
#include <SofaSimulationTree/GNode.h>
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
    sofa::component::visualmodel::OglModel *visu;
    sofa::component::mapping::RigidMapping< Rigid3dTypes , ExtVec3fTypes  > *mapping;
} VisualComponent;

typedef struct
{
    VirtContext m_virtContext;
    MechanicalStateForceFeedback<Rigid3dTypes>* forceFeedback;
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

    virtual void init();
    virtual void reinit();
    virtual void bwdInit();
    virtual void reset();
    virtual void handleEvent(core::objectmodel::Event *);
    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *);
    void onAnimateBeginEvent();
    int initDevice(char* ip);
    void closeDevice();
    static void haptic_callback(VirtContext, void *);

    void setForceFeedback(MechanicalStateForceFeedback<Rigid3dTypes>* ff);

private:

    HaptionData myData;
    VirtIndexingType m_indexingMode;
    VirtCommandType m_typeCommand;
    float m_speedFactor;
    float m_forceFactor;
    float haptic_time_step;
    int connection_device;
    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *rigidDOF;
    bool initCallback;
    simulation::Node *nodeHaptionVisual;
    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *visualHaptionDOF;
    simulation::Node *nodeAxesVisual;
    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *visualAxesDOF;
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
