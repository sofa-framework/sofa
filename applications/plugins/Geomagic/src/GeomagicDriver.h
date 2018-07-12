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
#ifndef SOFA_GEOMAGIC_H
#define SOFA_GEOMAGIC_H

//Geomagic include
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>

#ifndef SOFA_NO_OPENGL
#include <SofaOpenglVisual/OglModel.h>
#endif

#include <SofaUserInteraction/Controller.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <SofaHaptics/ForceFeedback.h>

//force feedback
#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaSimulationTree/GNode.h>

#include <math.h>
//#include <wrapper.h>
#include <HD/hd.h>
#include <HD/hdDevice.h>
#include <HD/hdDefines.h>
#include <HD/hdExport.h>
#include <HD/hdScheduler.h>

//Visualization
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa {

namespace component {

namespace controller {

using namespace sofa::defaulttype;
using core::objectmodel::Data;

#define NBJOINT 6

/**
* Geomagic driver
*/
class GeomagicDriver : public Controller
{

public:
    SOFA_CLASS(GeomagicDriver, Controller);
    typedef RigidTypes::Coord Coord;
    typedef RigidTypes::VecCoord VecCoord;
    typedef SolidTypes<double>::Transform Transform;

    typedef defaulttype::Vec4f Vec4f;
    typedef defaulttype::Vector3 Vector3;
    struct VisualComponent
    {
        simulation::Node::SPtr node;
        sofa::component::visualmodel::OglModel::SPtr visu;
        sofa::component::mapping::RigidMapping< Rigid3dTypes , ExtVec3fTypes  >::SPtr mapping;
    };

    Data< std::string > d_deviceName; ///< Name of device Configuration
    Data<Vec3d> d_positionBase; ///< Position of the interface base in the scene world coordinates
    Data<Quat> d_orientationBase; ///< Orientation of the interface base in the scene world coordinates
    Data<Quat> d_orientationTool; ///< Orientation of the tool

    Data<Vector6> d_dh_theta; ///< Denavit theta
    Data<Vector6> d_dh_alpha; ///< Denavit alpha
    Data<Vector6> d_dh_d; ///< Denavit d
    Data<Vector6> d_dh_a; ///< Denavit a

    Data<Vector6> d_angle; ///< Angluar values of joint (rad)
    Data<double> d_scale; ///< Default scale applied to the Phantom Coordinates
    Data<double> d_forceScale; ///< Default forceScale applied to the force feedback. 
    Data<bool> d_frameVisu; ///< Visualize the frame corresponding to the device tooltip
    Data<bool> d_omniVisu; ///< Visualize the frame of the interface in the virtual scene
    Data< VecCoord > d_posDevice; ///< position of the base of the part of the device
    Data<bool> d_button_1; ///< Button state 1
    Data<bool> d_button_2; ///< Button state 2
    Data<Vector3> d_inputForceFeedback; ///< Input force feedback in case of no LCPForceFeedback is found (manual setting)
    Data<double> d_maxInputForceFeedback; ///< Maximum value of the normed input force feedback for device security

    GeomagicDriver();

	virtual ~GeomagicDriver();

    virtual void init() override;
    virtual void bwdInit() override;
    virtual void reinit() override;
    virtual void draw(const sofa::core::visual::VisualParams* vparams) override;
    void updatePosition();

    void onAnimateBeginEvent();

    ForceFeedback * m_forceFeedback;

    /// variable pour affichage graphique
    enum
    {
        VN_stylus = 0,
        VN_joint2 = 1,
        VN_joint1 = 2,
        VN_arm2   = 3,
        VN_arm1   = 4,
        VN_joint0 = 5,
        VN_base   = 6,
        NVISUALNODE = 7
    };
    VisualComponent visualNode[NVISUALNODE];
    static const char* visualNodeNames[NVISUALNODE];
    static const char* visualNodeFiles[NVISUALNODE];
    simulation::Node::SPtr nodePrincipal;
    component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::SPtr rigidDOF;
    bool m_visuActive; ///< Internal boolean to detect activation switch of the draw
    bool m_initVisuDone; ///< Internal boolean activated only if visu initialization done without return

private:
    void handleEvent(core::objectmodel::Event *) override;
    void computeBBox(const core::ExecParams*  params, bool onlyVisible=false ) override;
    void getMatrix( Mat<4,4, GLdouble> & M, int index, double teta);

    Mat<4,4, GLdouble> compute_dh_Matrix(double theta,double alpha, double a, double d);

    Mat<4,4, GLdouble> m_dh_matrices[NBJOINT];

    //These data are written by the omni they cnnot be accessed in the simulation loop
    struct OmniData
    {
        HDdouble angle1[3];
        HDdouble angle2[3];
        HDdouble transform[16];
        int buttonState;
    };

public:
    OmniData m_omniData;
    OmniData m_simuData;
    HHD m_hHD;
    std::vector< HDCallbackCode > m_hStateHandles;

};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_GEOMAGIC_H
