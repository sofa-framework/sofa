/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_ODESOLVER_OMNISOLVER_H
#define SOFA_COMPONENT_ODESOLVER_OMNISOLVER_H

//Sensable include
#include <HD/hd.h>
#include <HDU/hdu.h>
#include <HDU/hduError.h>
#include <HDU/hduVector.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>

#include <sofa/core/behavior/BaseController.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaUserInteraction/Controller.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/visual/VisualParams.h>

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
struct DeviceData
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

struct OmniData
{
    vector<ForceFeedback*> forceFeedbacks;
    int forceFeedbackIndice;
    simulation::Node *context;

    sofa::defaulttype::SolidTypes<double>::Transform endOmni_H_virtualTool;
    //Transform baseOmni_H_endOmni;
    sofa::defaulttype::SolidTypes<double>::Transform world_H_baseOmni;
    double forceScale;
    double scale;
    bool permanent_feedback;

    // API OMNI //
    DeviceData servoDeviceData;  // for the haptic loop
    DeviceData deviceData;		 // for the simulation loop

};

/**
* Omni driver
*/
class OmniDriver : public Controller
{

public:
    SOFA_CLASS(OmniDriver, Controller);
    Data<double> scale;
    Data<double> forceScale;
    Data<Vec3d> positionBase;
    Data<Quat> orientationBase;
    Data<Vec3d> positionTool;
    Data<Quat> orientationTool;
    Data<bool> permanent;
    Data<bool> omniVisu;
    Data<bool> toolSelector;
    Data<int> toolCount;

    OmniData	data;

    OmniDriver();
    virtual ~OmniDriver();

    virtual void init();
    virtual void bwdInit();
    virtual void reset();
    void reinit();

    int initDevice(OmniData& data);

    void cleanup();
    virtual void draw();
	virtual void draw(const core::visual::VisualParams*) override;

    void setForceFeedbacks(vector<ForceFeedback*> ffs);

    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *);

    void setDataValue();
    void reinitVisual();

private:
    void handleEvent(core::objectmodel::Event *);
    sofa::component::visualmodel::OglModel::SPtr visu_base, visu_end;
    bool noDevice;

    bool moveOmniBase;
    Vec3d positionBase_buf;
    core::behavior::MechanicalState<Rigid3dTypes> *mState; ///< Controlled MechanicalState.

    int currentToolIndex;
    bool isToolControlled;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ODESOLVER_OMNISOLVER_H
