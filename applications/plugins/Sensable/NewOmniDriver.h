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

//Sensable include
#include <HD/hd.h>
#include <HDU/hdu.h>
#include <HDU/hduError.h>
#include <HDU/hduVector.h>
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


//force feedback
#include <SofaHaptics/ForceFeedback.h>
#include <SofaHaptics/MechanicalStateForceFeedback.h>
#include <SofaHaptics/LCPForceFeedback.h>
#include <SofaHaptics/NullForceFeedbackT.h>

#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaOpenglVisual/OglModel.h>
#include <SofaSimulationTree/GNode.h>
#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseVisual/InteractiveCamera.h>

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
using namespace sofa::helper;
using core::objectmodel::Data;

/** Holds data retrieved from HDAPI. */
struct NewDeviceData
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
};

struct NewOmniData
{
    ForceFeedback::SPtr forceFeedback;
    simulation::Node::SPtr *context;

    sofa::defaulttype::SolidTypes<double>::Transform endOmni_H_virtualTool;
    //Transform baseOmni_H_endOmni;
    sofa::defaulttype::SolidTypes<double>::Transform world_H_baseOmni;
    double forceScale;
    double scale;
    bool permanent_feedback;

    // API OMNI //
    NewDeviceData servoDeviceData;  // for the haptic loop
    NewDeviceData deviceData;		 // for the simulation loop

    double currentForce[3];

};

struct AllNewOmniData
{
    vector<NewOmniData> omniData;
} ;

/**
* Omni driver
*/
class NewOmniDriver : public Controller
{

public:
    SOFA_CLASS(NewOmniDriver, Controller);
    typedef RigidTypes::Coord Coord;
    typedef RigidTypes::VecCoord VecCoord;
    typedef component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> MMechanicalObject;



    struct VisualComponent
    {
        simulation::Node::SPtr node;
        sofa::component::visualmodel::OglModel::SPtr visu;
        sofa::component::mapping::RigidMapping< Rigid3dTypes , ExtVec3fTypes  >::SPtr mapping;
    };



    Data<double> forceScale; ///< Default forceScale applied to the force feedback. 
    Data<double> scale; ///< Default scale applied to the Phantom Coordinates. 
    Data<Vec3d> positionBase; ///< Position of the interface base in the scene world coordinates
    Data<Quat> orientationBase; ///< Orientation of the interface base in the scene world coordinates
    Data<Vec3d> positionTool; ///< Position of the tool in the omni end effector frame
    Data<Quat> orientationTool; ///< Orientation of the tool in the omni end effector frame
    Data<bool> permanent; ///< Apply the force feedback permanently
    Data<bool> omniVisu; ///< Visualize the position of the interface in the virtual scene
    Data< VecCoord > posDevice; ///< position of the base of the part of the device
    Data< VecCoord > posStylus; ///< position of the base of the stylus
    Data< std::string > locDOF; ///< localisation of the DOFs MechanicalObject
    Data< std::string > deviceName; ///< name of the device
    Data< int > deviceIndex; ///< index of the device
    Data<Vec1d> openTool; ///< opening of the tool
    Data<double> maxTool; ///< maxTool value
    Data<double> minTool; ///< minTool value
    Data<double> openSpeedTool; ///< openSpeedTool value
    Data<double> closeSpeedTool; ///< closeSpeedTool value
    Data<bool> useScheduler; ///< Enable use of OpenHaptics Scheduler methods to synchronize haptics thread
    Data<bool> setRestShape; ///< True to control the rest position instead of the current position directly
    Data<bool> applyMappings; ///< True to enable applying the mappings after setting the position
    Data<bool> alignOmniWithCamera; ///< True to keep the Omni's movements in the same reference frame as the camera
	Data<bool> stateButton1; ///< True if the First button of the Omni is pressed
	Data<bool> stateButton2; ///< True if the Second button of the Omni is pressed



    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *DOFs;
    sofa::component::visualmodel::BaseCamera::SPtr camera;

    bool initVisu;

    NewOmniData data;
    AllNewOmniData allData;

    NewOmniDriver();
    virtual ~NewOmniDriver();

    virtual void init();
    virtual void bwdInit();
    virtual void reset();
    void reinit();

    int initDevice();

    void cleanup();
	virtual void draw(const core::visual::VisualParams*) override;
    virtual void draw();

    void setForceFeedback(ForceFeedback* ff);

    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *);
    void onAnimateBeginEvent();

    void setDataValue();

    //variable pour affichage graphique
    enum
    {
        VN_stylus = 0,
        VN_joint2 = 1,
        VN_joint1 = 2,
        VN_arm2   = 3,
        VN_arm1   = 4,
        VN_joint0 = 5,
        VN_base   = 6,
        VN_X      = 7,
        VN_Y      = 8,
        VN_Z      = 9,
        NVISUALNODE = 10
    };
    VisualComponent visualNode[NVISUALNODE];
    static const char* visualNodeNames[NVISUALNODE];
    static const char* visualNodeFiles[NVISUALNODE];
    simulation::Node::SPtr nodePrincipal;
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
