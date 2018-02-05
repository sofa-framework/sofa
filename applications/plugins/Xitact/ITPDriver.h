/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ITPDRIVER_H
#define SOFA_COMPONENT_ITPDRIVER_H

//Sensable include
#include <sofa/core/VecId.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/BaseController.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaUserInteraction/Controller.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/Quat.h>
#include "XiTrocarInterface.h"

//#include <sofa/gui/PickHandler.h>
//#include <sofa/gui/qt/SofaMouseManager.h>
#include <SofaUserInteraction/RayModel.h>
#include <Xitact/config.h>

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

typedef SOFA_XITACTPLUGIN_API struct
{
    ForceFeedback* forceFeedback;
    simulation::Node *context;

    double scale;
    double forceScale;
    bool permanent_feedback;

    // API ITP //
    XiToolState hapticState;     // for the haptic loop
    XiToolState simuState;		 // for the simulation loop
    XiToolForce hapticForce;

} XiToolDataITP;

/**
* ITP Xitact driver
* http://www.mentice.com/default.asp?viewset=1&on=%27Products%27&id=&initid=99&heading=Products&mainpage=templates/05.asp?sida=85
*/
class SOFA_XITACTPLUGIN_API ITPDriver : public sofa::component::controller::Controller
{

public:

    SOFA_CLASS(ITPDriver,sofa::component::controller::Controller);
    Data<double> Scale;
    Data<bool> permanent;
    Data <int> indexTool;
    Data <sofa::defaulttype::Vec3d> direction;
    Data <sofa::defaulttype::Vec3d> position;

    XiToolDataITP	data;

    ITPDriver();
    virtual ~ITPDriver();

    virtual void bwdInit();
    virtual void reset();
    void reinit();

    void cleanup();
    //void draw();

    void setForceFeedback(ForceFeedback* ff);


    void updateForce();
    void setDataValue();
    void reinitVisual();
    float graspReferencePoint[3];
    bool contactReached;
    float ToolD;
private:
    sofa::core::behavior::MechanicalState<Vec1dTypes> *_mstate;
    void handleEvent(core::objectmodel::Event *);
    sofa::component::visualmodel::OglModel *visu_base, *visu_end;
    bool noDevice;
    Quat fromGivenDirection( Vector3& dir,  Vector3& local_dir, Quat old_quat);

    XiToolState restState;

    void mainButtonPushed();
    void rightButtonPushed();
    void leftButtonPushed();


    //sofa::component::collision::HeartSimulationManager* heartManager;

    bool operation; // true = right, false = left




};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ITPDRIVER_H
