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
#ifndef SOFA_COMPONENT_IHPDRIVER_H
#define SOFA_COMPONENT_IHPDRIVER_H

#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/BaseController.h>
#include <SofaUserInteraction/Controller.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/Quat.h>
#include "XiTrocarInterface.h"
#include <SofaHaptics/LCPForceFeedback.h>
#include <SofaHaptics/MechanicalStateForceFeedback.h>
#include <SofaHaptics/NullForceFeedbackT.h>
#include <sofa/simulation/Node.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaRigid/RigidMapping.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#ifdef SOFA_DEV
#include <sofa/component/controller/VMechanismsForceFeedback.h>
#endif
#include "PaceMaker.h"
#include <Xitact/config.h>
namespace sofa
{
namespace simulation { class Node; }

namespace component
{
namespace visualModel { class OglModel; }

namespace controller
{

using namespace sofa::defaulttype;
using core::objectmodel::Data;

// Force FeedBack safety threshold, manually observed in Xitact example. Change it if needed.
static float FFthresholdX = 30.0; //in Newtons
static float FFthresholdY = 30.0;
static float FFthresholdZ = 30.0;
static float FFthresholdRoll;


typedef struct
{
    //MechanicalStateForceFeedback<Rigid3dTypes>* lcp_forceFeedback;//= NULL;
    LCPForceFeedback<Rigid3dTypes>* lcp_forceFeedback;//= NULL;
#ifdef SOFA_DEV
    VMechanismsForceFeedback<defaulttype::Vec1dTypes>* vm_forceFeedback;// = NULL;
#endif
    simulation::Node *context;

    int indexTool;
    double scale;
    double forceScale;
    bool permanent_feedback;
    bool lcp_true_vs_vm_false;

    // API IHP //
    XiToolState hapticState;     // for the haptic loop
    XiToolState simuState;		 // for the simulation loop
    XiToolState restState;       // for initial haptic state
    XiToolForce hapticForce;

    //RigidTypes::VecCoord positionBaseGlobal;

    Vec3d posBase;
    Quat quatBase;


} XiToolDataIHP;

typedef struct
{
    vector<XiToolDataIHP*>  xiToolData;
} allXiToolDataIHP;

typedef struct
{
    simulation::Node::SPtr node;
    sofa::component::visualmodel::OglModel::SPtr visu;
    sofa::component::mapping::RigidMapping< Rigid3dTypes , ExtVec3fTypes  >::SPtr mapping;

} VisualComponent;


/**
* IHP Xitact driver
* http://www.mentice.com/default.asp?viewset=1&on=%27Products%27&id=&initid=99&heading=Products&mainpage=templates/05.asp?sida=85
*/
class SOFA_XITACTPLUGIN_API IHPDriver : public sofa::component::controller::Controller
{


public:
    SOFA_CLASS(IHPDriver,sofa::component::controller::Controller);
    typedef RigidTypes::VecCoord VecCoord;

    Data<double> Scale; ///< Default scale applied to the Phantom Coordinates. 
    Data<double> forceScale; ///< Default scale applied to the force feedback. 
    Data<bool> permanent; ///< Apply the force feedback permanently
    Data<int> indexTool; ///< index of the tool to simulate (if more than 1). Index 0 correspond to first tool.
    Data<double> graspThreshold; ///< Threshold value under which grasping will launch an event.
    Data<bool> showToolStates; ///< Display states and forces from the tool.
    Data<bool> testFF; ///< If true will add force when closing handle. As if tool was entering an elastic body.
    Data<int> RefreshFrequency; ///< Frequency of the haptic loop.
    Data<bool> xitactVisu; ///< Visualize the position of the interface in the virtual scene
    Data< VecCoord > positionBase; ///< position of the base of the device
    Data<string> locPosBati; ///< localisation of the restPosition of the bati
    Data<int> deviceIndex; ///< index of the device
    Data<Vec1d> openTool; ///< opening of the tool
    Data<double> maxTool; ///< maxTool value
    Data<double> minTool; ///< minTool value

    allXiToolDataIHP allData;
    XiToolDataIHP data;

    vector<IHPDriver*> otherXitact;

    IHPDriver();
    virtual ~IHPDriver();
#ifdef XITACT_VISU
    virtual void init();
#endif
    virtual void bwdInit();
    virtual void reset();
    void reinit();

    void cleanup();
    //virtual void draw();

    void setLCPForceFeedback(LCPForceFeedback<Rigid3dTypes>* ff);
#ifdef SOFA_DEV
    void setVMForceFeedback(VMechanismsForceFeedback<defaulttype::Vec1dTypes>* ff);
#endif

    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *);

    void createCallBack();
    void deleteCallBack();
    void stateCallBack();

    void updateForce();
    void displayState();


    void setDataValue();
    void reinitVisual();

    double getScale () {return Scale.getValue();};

    void rightButtonPushed();
    void leftButtonPushed();
    void graspClosed();

    bool operation; // true = right, false = left


private:
    void handleEvent(core::objectmodel::Event *);
    sofa::component::visualmodel::OglModel *visu_base, *visu_end;
    bool noDevice;
    Quat fromGivenDirection( Vector3& dir,  Vector3& local_dir, Quat old_quat);

    bool graspElasticMode;
    sofa::component::controller::PaceMaker* myPaceMaker;

    bool findForceFeedback;

    bool initVisu;
    bool changeScale;
    float oldScale;
    VisualComponent visualNode[7];
    simulation::Node::SPtr nodePrincipal;
    simulation::Node::SPtr nodeBati;
    simulation::Node::SPtr nodeAxes;
    simulation::Node::SPtr nodeTool;
    simulation::Node::SPtr nodeAxesVisual;
    simulation::Node::SPtr nodeXitactVisual;
    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::SPtr visualAxesDOF;
    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::SPtr visualXitactDOF;


    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *posBati;
    //sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *rigidDOF;
    //sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *axes;
    sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> *posTool;
    bool visuActif;
    bool visuAxes;
    bool modX,modY,modZ,modS;
    bool firstDevice;





};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_IHPDRIVER_H
