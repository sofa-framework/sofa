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
#ifndef SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H
#define SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H

//Sensable include
#include <HD/hd.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>

#include <sofa/core/componentmodel/behavior/BaseController.h>
#include <sofa/component/visualModel/OglModel.h>
#include <sofa/component/controller/Controller.h>

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
    // hduVector3Dd m_devicePosition;	/* Current device coordinates. */
    HDErrorInfo m_error;
    Vec3d pos;
    Quat quat;
    bool ready;
    bool stop;
} DeviceData;

typedef struct
{
    ForceFeedback* forceFeedback;
    simulation::Node *context;

    sofa::defaulttype::SolidTypes<double>::Transform endOmni_H_virtualTool;
    //Transform baseOmni_H_endOmni;
    sofa::defaulttype::SolidTypes<double>::Transform world_H_baseOmni;
    double scale;
    double forceScale;
    bool permanent_feedback;

    // API OMNI //
    DeviceData servoDeviceData;  // for the haptic loop
    DeviceData deviceData;		 // for the simulation loop

} OmniData;

/**
* Omni driver
*/
class NewOmniDriver : public Controller
{

public:
    Data<double> Scale;
    Data<double> forceScale;
    Data<Vec3d> positionBase;
    Data<Quat> orientationBase;
    Data<Vec3d> positionTool;
    Data<Quat> orientationTool;
    Data<bool> permanent;
    Data<bool> OmniVisu;

    OmniData	data;

    NewOmniDriver();
    virtual ~NewOmniDriver();

    virtual void bwdInit();
    virtual void reset();
    void reinit();

    void cleanup();
    virtual void draw();

    void setForceFeedback(ForceFeedback* ff);

    void onKeyPressedEvent(core::objectmodel::KeypressedEvent *);
    void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *);

    void setDataValue();
    void reinitVisual();

private:
    void handleEvent(core::objectmodel::Event *);
    sofa::component::visualmodel::OglModel *visu_base, *visu_end;
    bool noDevice;

    bool moveOmniBase;
    Vec3d positionBase_buf;




};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ODESOLVER_NEWOMNISOLVER_H
