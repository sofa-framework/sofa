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
#pragma once

//Geomagic include
#include <Geomagic/config.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>
#include <SofaUserInteraction/Controller.h>

//force feedback
#include <SofaHaptics/ForceFeedback.h>

#include <HD/hd.h>

namespace sofa::component::controller
{

using namespace sofa::defaulttype;


class GeomagicVisualModel;


/**
* Class providing a driver API to handle Geomagic haptic device and the servo loop scheduler.
* Position and angles of the device will be retrieve inside @sa m_hapticData during a callback method executed by HD scheduler.
* This callback is also setting force to the device if a ForceFeedBack component is used in the simulation scene.
* Those data are then copied inside @sa m_simuData and used to update Device position in SOFA world in method @sa updatePosition()
*/
class SOFA_GEOMAGIC_API GeomagicDriver : public Controller
{

public:
    SOFA_CLASS(GeomagicDriver, Controller);
    typedef RigidTypes::Coord Coord;
    typedef RigidTypes::VecCoord VecCoord;
    
    GeomagicDriver();
    virtual ~GeomagicDriver();

    // SOFA component API override
    void init() override;
    void draw(const sofa::core::visual::VisualParams* vparams) override;

    void handleEvent(core::objectmodel::Event *) override;
    void computeBBox(const core::ExecParams*  params, bool onlyVisible = false) override;

    /// Public method to init tool. Can be called from thirdparty if @sa d_manualStart is set to true
    virtual void initDevice();
    
    /// Method to clear sheduler and free device. Called by default at driver destruction
    virtual void clearDevice();


protected:
    /// Internal method to update the position of the device in SOFA world using @sa m_simuData 
    void updatePosition();

    /** Internal method called by @sa updatePosition() to handle the button status part using as well @sa m_simuData
    * if @sa d_emitButtonEvent is set to true, event will be fired by this method if button is pressed or released
    */
    void updateButtonStates();
    

public:
    //Input Data
    Data< std::string > d_deviceName; ///< Name of device Configuration
    Data<Vec3d> d_positionBase; ///< Input Position of the device base in the scene world coordinates
    Data<Quat> d_orientationBase; ///< Input Orientation of the device base in the scene world coordinates
    Data<Quat> d_orientationTool; ///< Input Orientation of the tool
    Data<SReal> d_scale; ///< Default scale applied to the device Coordinates
    Data<SReal> d_forceScale; ///< Default forceScale applied to the force feedback. 
    Data<SReal> d_maxInputForceFeedback; ///< Maximum value of the normed input force feedback for device security
    Data<Vector3> d_inputForceFeedback; ///< Input force feedback in case of no LCPForceFeedback is found (manual setting)

    // Input parameters
    Data<bool> d_manualStart; /// < Bool to unactive the automatic start of the device at init. initDevice need to be called manually. False by default.
    Data<bool> d_emitButtonEvent; ///< Bool to send event through the graph when button are pushed/released
    Data<bool> d_frameVisu; ///< Visualize the frame corresponding to the device tooltip
    Data<bool> d_omniVisu; ///< Visualize the frame of the interface in the virtual scene

    //Output Data
    Data<Coord> d_posDevice; ///< position of the base of the part of the device
    Data<Vector6> d_angle; ///< Angluar values of joint (rad)
    Data<bool> d_button_1; ///< Button state 1
    Data<bool> d_button_2; ///< Button state 2
    
    // Pointer to the forceFeedBack component
    ForceFeedback::SPtr m_forceFeedback;
    // link to the forceFeedBack component, if not set will search through graph and take first one encountered
    SingleLink<GeomagicDriver, ForceFeedback, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_forceFeedback;

    /// This static bool is used to know if HD scheduler is already running. No mechanism provided by Hd lib.
    inline static bool s_schedulerRunning = false;

protected:
    // Pointer to the Geomagic visual model to draw device in scene
    std::unique_ptr<GeomagicVisualModel> m_GeomagicVisualModel;
   
public:
    ///These data are written by the omni they cnnot be accessed in the simulation loop
    struct DeviceData
    {
        HDdouble angle1[3];
        HDdouble angle2[3];
        HDdouble transform[16];
        int buttonState;
    };

    // Public members exchanged between Driver and HD scheduler
    bool m_simulationStarted; ///< Boolean to warn scheduler when SOFA has started the simulation (changed by AnimateBeginEvent)
    bool m_isInContact; ///< Boolean to warn SOFA side when scheduler has computer contact (forcefeedback no null)
    DeviceData m_hapticData; ///< data structure used by scheduler
    DeviceData m_simuData; ///< data structure used by SOFA loop, values are copied from @sa m_hapticData
    HHD m_hHD; ///< ID the device
    std::vector< HDSchedulerHandle > m_hStateHandles; ///< List of ref to the workers scheduled
};

} // namespace sofa::component::controller
