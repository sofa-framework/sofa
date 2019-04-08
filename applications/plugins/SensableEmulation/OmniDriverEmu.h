/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_OMNIEMU_H
#define SOFA_COMPONENT_CONTROLLER_OMNIEMU_H

#include <SofaUserInteraction/Controller.h>
#include <SofaOpenglVisual/OglModel.h>

#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/SolidTypes.h>

#include <sofa/helper/system/thread/CTime.h>

#ifndef WIN32
#  include <pthread.h>
#endif

#include <SensableEmulation/config.h>


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

/** Holds data retrieved from HDAPI. */
typedef struct
{
    int nupdates;
    int m_buttonState;					/* Has the device button has been pressed. */
    Vec3d pos;
    Quat quat;
    bool ready;
    bool stop;
} DeviceData;

typedef struct
{
    helper::vector<ForceFeedback*> forceFeedbacks;
    int forceFeedbackIndice;
    simulation::Node *context;

    sofa::defaulttype::SolidTypes<double>::Transform endOmni_H_virtualTool;
    sofa::defaulttype::SolidTypes<double>::Transform world_H_baseOmni;
    double forceScale;
    double scale;
    bool permanent_feedback;

    // API OMNI //
    DeviceData servoDeviceData;  // for the haptic loop
    DeviceData deviceData;		 // for the simulation loop

} OmniData;

/**
* Omni driver emulator you can add to your scene.
*
* Controller's actions:
*  key z: reset to base position
*  key k, l, m: move base position
*  key h: emulate button 1 press/release
*  key i: emulate button 2 press/release
*/
class SOFA_SENSABLEEMUPLUGIN_API OmniDriverEmu : public Controller
{

public:
    typedef Rigid3dTypes::Coord Coord;
    typedef Rigid3dTypes::VecCoord VecCoord;

    SOFA_CLASS(OmniDriverEmu, Controller);
    Data<double> forceScale; ///< Default forceScale applied to the force feedback.
    Data<double> scale; ///< Default scale applied to the Phantom Coordinates.
    Data<Vec3d> positionBase; ///< Position of the interface base in the scene world coordinates
    Data<Quat> orientationBase; ///< Orientation of the interface base in the scene world coordinates
    Data<Vec3d> positionTool; ///< Position of the tool in the omni end effector frame
    Data<Quat> orientationTool; ///< Orientation of the tool in the omni end effector frame
    Data<bool> permanent; ///< Apply the force feedback permanently
    Data<bool> omniVisu; ///< Visualize the position of the interface in the virtual scene
    Data<int> simuFreq; ///< frequency of the "simulated Omni"
    Data<bool> simulateTranslation; ///< do very naive "translation simulation" of omni, with constant orientation <0 0 0 1>
    Data<bool> toolSelector;
    Data<size_t> toolCount;

    OmniData	data;

    OmniDriverEmu();
    ~OmniDriverEmu() override;

    void init() override;
    void bwdInit() override;
    void reset() override;
    void reinit() override;
    void cleanup() override;
    void draw(const core::visual::VisualParams*) override;

    int initDevice(OmniData& data);
    void setForceFeedbacks(helper::vector<ForceFeedback*> ffs);

    void setDataValue();

    void setOmniSimThreadCreated(bool b) { omniSimThreadCreated = b;}

    bool afterFirstStep;
    SolidTypes<double>::Transform prevPosition;

    //need for "omni simulation"
    helper::system::thread::CTime *thTimer;

#ifndef WIN32
    pthread_t hapSimuThread;
#endif

    double lastStep;
    bool executeAsynchro;
    Data<VecCoord> trajPts; ///< Trajectory positions
    Data<helper::vector<double> > trajTim; ///< Trajectory timing

    int getCurrentToolIndex() { return currentToolIndex;}
    void handleEvent(core::objectmodel::Event *) override ;

private:

    void copyDeviceDataCallback(OmniData *pUserData);
    void stopCallback(OmniData *pUserData);
    component::visualmodel::OglModel::SPtr visu_base, visu_end;
    bool noDevice;

    bool moveOmniBase;
    Vec3d positionBase_buf;

    core::behavior::MechanicalState<Rigid3dTypes> *mState; ///< Controlled MechanicalState.

    bool omniSimThreadCreated;
    int currentToolIndex;
    bool isToolControlled;
};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_OMNIEMU_H
