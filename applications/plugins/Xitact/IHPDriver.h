/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_IHPDRIVER_H
#define SOFA_COMPONENT_IHPDRIVER_H


#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/component/visualModel/OglModel.h>
#include <sofa/component/controller/Controller.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/Quat.h>
#include "XiTrocarInterface.h"
#include <sofa/component/controller/LCPForceFeedback.h>
#ifdef SOFA_DEV
#include <sofa/component/controller/VMechanismsForceFeedback.h>
#endif
#include "PaceMaker.h"
#include "initXitact.h"
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
    LCPForceFeedback<defaulttype::Vec1dTypes>* lcp_forceFeedback;//= NULL;
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

} XiToolDataIHP;


/**
* IHP Xitact driver
* http://www.mentice.com/default.asp?viewset=1&on=%27Products%27&id=&initid=99&heading=Products&mainpage=templates/05.asp?sida=85
*/
class SOFA_XITACTPLUGIN_API IHPDriver : public sofa::component::controller::Controller
{


public:
    SOFA_CLASS(IHPDriver,sofa::component::controller::Controller);

    Data<double> Scale;
    Data<double> forceScale;
    Data<bool> permanent;
    Data<int> indexTool;
    Data<double> graspThreshold;
    Data<bool> showToolStates;
    Data<bool> testFF;
    Data<int> RefreshFrequency;


    XiToolDataIHP	data;

    IHPDriver();
    virtual ~IHPDriver();

    virtual void bwdInit();
    virtual void reset();
    void reinit();

    void cleanup();
    //virtual void draw();

    void setLCPForceFeedback(LCPForceFeedback<defaulttype::Vec1dTypes>* ff);
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
    sofa::core::behavior::MechanicalState<Vec1dTypes> *_mstate;
    void handleEvent(core::objectmodel::Event *);
    sofa::component::visualmodel::OglModel *visu_base, *visu_end;
    bool noDevice;
    Quat fromGivenDirection( Vector3& dir,  Vector3& local_dir, Quat old_quat);

    bool graspElasticMode;
    sofa::component::controller::PaceMaker* myPaceMaker;

    bool findForceFeedback;




};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_IHPDRIVER_H
