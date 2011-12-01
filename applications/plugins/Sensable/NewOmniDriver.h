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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H
#define SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H

//Sensable include
#include "HD/hd.h"
#include "HDU/hdu.h"
#include "HDU/hduError.h"
#include "HDU/hduVector.h"
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>


#include <sofa/core/behavior/BaseController.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/controller/Controller.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/container/MechanicalObject.h>

//truc du .cpp
#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/objectmodel/HapticDeviceEvent.h>

//force feedback
//#include <sofa/component/controller/ForceFeedback.h>
#include <sofa/component/controller/MechanicalStateForceFeedback.h>
#include <sofa/component/controller/LCPForceFeedback.h>
#include <sofa/component/controller/NullForceFeedbackT.h>

#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <sofa/simulation/common/Node.h>
#include <cstring>

#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/simulation/tree/GNode.h>

#include <math.h>

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

/** Holds data retrieved from HDAPI. */
typedef struct
{
    HHD id;
    int nupdates;
    int m_buttonState;					/* Has the device button has been pressed. */
    hduVector3Dd m_devicePosition;	/* Current device coordinates. */
    HDErrorInfo m_error;
    Vec3d pos;
    Quat quat;
    bool ready;
    bool stop;
} DeviceData;

typedef struct
{
    simulation::Node *node;
    sofa::component::visualmodel::OglModel::SPtr visu;
    sofa::component::mapping::RigidMapping< Rigid3dTypes , ExtVec3fTypes  >::SPtr mapping;

} VisualComponent;

typedef struct
{
    LCPForceFeedback<Rigid3dTypes>::SPtr forceFeedback;
    simulation::Node::SPtr *context;

    sofa::defaulttype::SolidTypes<double>::Transform endOmni_H_virtualTool;
    //Transform baseOmni_H_endOmni;
    sofa::defaulttype::SolidTypes<double>::Transform world_H_baseOmni;
    double forceScale;
    double scale;
    bool permanent_feedback;

    // API OMNI //
    DeviceData servoDeviceData;  // for the haptic loop
    DeviceData deviceData;		 // for the simulation loop

    double currentForce[3];

} OmniData;

typedef struct
{
    vector<OmniData> omniData;
} allOmniData;

/**
* Omni driver
*/
class NewOmniDriver : public Controller
{

public:
    SOFA_CLASS(NewOmniDriver, Controller);
    typedef RigidTypes::VecCoord Coord;
    typedef RigidTypes::VecCoord VecCoord;
    typedef component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> MMechanicalObject;





    Data<double> forceScale;
    Data<double> scale;
    Data<Vec3d> positionBase;
    Data<Quat> orientationBase;
    Data<Vec3d> positionTool;
    Data<Quat> orientationTool;
    Data<bool> permanent;
    Data<bool> omniVisu;
    Data< VecCoord > posDevice;
    Data< VecCoord > posStylus;
    Data< std::string > locDOF;
    Data< std::string > deviceName;
    Data< int > deviceIndex;
    Data<Vec1d> openTool;
    Data<double> maxTool;
    Data<double> minTool;
    Data<double> openSpeedTool;
    Data<double> closeSpeedTool;

    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *DOFs;

    bool initVisu;

    OmniData data;
    allOmniData allData;

    NewOmniDriver();
    virtual ~NewOmniDriver();

    virtual void init();
    virtual void bwdInit();
    virtual void reset();
    void reinit();

    int initDevice();

    void cleanup();
    virtual void draw();

    void setForceFeedback(LCPForceFeedback<Rigid3dTypes>* ff);

    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *);
    void onAnimateBeginEvent();

    void setDataValue();

    //variable pour affichage graphique
    simulation::Node *parent;
    VisualComponent visualNode[10];
    simulation::Node *nodePrincipal;
    simulation::Node *nodeDOF;
    MMechanicalObject::SPtr rigidDOF;
    bool changeScale;
    bool firstInit;
    float oldScale;
    bool visuActif;
    bool isInitialized;
    Vec3d positionBase_buf;
    bool modX;
    bool modY;
    bool modZ;
    bool modS;
    bool axesActif;
    double pi;
    HDfloat angle1[3];
    HDfloat angle2[3];
    bool firstDevice;
    //vector<NewOmniDriver*> autreOmniDriver;

private:
    void handleEvent(core::objectmodel::Event *);
    bool noDevice;



};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H
