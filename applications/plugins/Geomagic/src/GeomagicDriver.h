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
#ifndef SOFA_GEOMAGIC_H
#define SOFA_GEOMAGIC_H

//Sensable include
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

    Data< std::string > d_deviceName;
    Data<Vec3d> d_positionBase;
    Data<Quat> d_orientationBase;
    Data<Quat> d_orientationTool;

    Data<Vector6> d_dh_theta;
    Data<Vector6> d_dh_alpha;
    Data<Vector6> d_dh_d;
    Data<Vector6> d_dh_a;

    Data<Vector6> d_angle;
    Data<double> d_scale;
    Data<double> d_forceScale;
    Data<bool> d_omniVisu;
    Data< Coord > d_posDevice;
    Data<bool> d_button_1;
    Data<bool> d_button_2;

    GeomagicDriver();

	virtual ~GeomagicDriver();

    virtual void init();
    virtual void bwdInit();
    virtual void reinit();
    virtual void draw(const sofa::core::visual::VisualParams* vparams);
    void updatePosition();

    void onAnimateBeginEvent();

    ForceFeedback * m_forceFeedback;

private:
    void handleEvent(core::objectmodel::Event *);
    void computeBBox(const core::ExecParams*  params, bool onlyVisible=false );
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
