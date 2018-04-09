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

#include "GeomagicDriver.h"
#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/visual/VisualParams.h>


namespace sofa
{

namespace component
{

namespace controller
{


static Mat<4,4, GLdouble> getInitialTransform() {
    Mat<4,4, GLdouble> M;

    const sofa::defaulttype::SolidTypes<double>::Transform transformOmni1(Vector3(0,0,0),Quat(Vector3(1,0,0),-M_PI/2));
    const sofa::defaulttype::SolidTypes<double>::Transform transformOmni2(Vector3(0,0,2.3483),Quat(Vector3(0,0,1),-M_PI/2));
    const sofa::defaulttype::SolidTypes<double>::Transform transformOmni3(Vector3(-16.8473,0,0),Quat(Vector3(1,0,0),M_PI));

    sofa::defaulttype::SolidTypes<double>::Transform T = transformOmni1*transformOmni2*transformOmni3;
    T.writeOpenGlMatrix((double*) M.ptr());

    return M;
}

//change the axis of the omni
static const Mat<4,4, GLdouble> initialTransform = getInitialTransform();

using namespace sofa::defaulttype;

void printError(const HDErrorInfo *error, const char *message)
{
    std::cerr << hdGetErrorString(error->errorCode) << std::endl;
    std::cerr << "HHD: "<< error->hHD << std::endl;
    std::cerr << "Error Code: "<< error->hHD << std::endl;
    std::cerr << "Internal Error Code: "<< error->internalErrorCode<<std::endl;
    std::cerr << "Message: " << message << std::endl;
}

HDCallbackCode HDCALLBACK copyDeviceDataCallback(void * pUserData)
{
    GeomagicDriver * driver = (GeomagicDriver * ) pUserData;
    driver->m_simuData = driver->m_omniData;
	return HD_CALLBACK_CONTINUE;
}

HDCallbackCode HDCALLBACK stateCallback(void * userData)
{
    HDErrorInfo error;
    GeomagicDriver * driver = (GeomagicDriver * ) userData;

    hdMakeCurrentDevice(driver->m_hHD);
    if (HD_DEVICE_ERROR(error = hdGetError())) return HD_CALLBACK_CONTINUE;

    hdBeginFrame(driver->m_hHD);
    if (HD_DEVICE_ERROR(error = hdGetError())) return HD_CALLBACK_CONTINUE;

    hdGetIntegerv(HD_CURRENT_BUTTONS, &driver->m_omniData.buttonState);
    hdGetDoublev(HD_CURRENT_TRANSFORM, driver->m_omniData.transform);

    //angles
    hdGetDoublev(HD_CURRENT_JOINT_ANGLES,driver->m_omniData.angle1);
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES,driver->m_omniData.angle2);

    if (driver->m_forceFeedback) {
        Vector3 currentForce;
        Vector3 currentForced;
        currentForced.clear();
        Vector3 pos(driver->m_omniData.transform[12+0]*0.1,driver->m_omniData.transform[12+1]*0.1,driver->m_omniData.transform[12+2]*0.1);

        Vector3 pos_in_world = driver->d_positionBase.getValue() + driver->d_orientationBase.getValue().rotate(pos*driver->d_scale.getValue());

        driver->m_forceFeedback->computeForce(pos_in_world[0],pos_in_world[1],pos_in_world[2], 0, 0, 0, 0, currentForce[0], currentForce[1], currentForce[2]);

        Vector3 force_in_omni = driver->d_orientationBase.getValue().inverseRotate(currentForce)  * driver->d_forceScale.getValue();

        double omni_force[3];
        omni_force[0] = force_in_omni[0];
        omni_force[1] = force_in_omni[1];
        omni_force[2] = force_in_omni[2];

        hdSetDoublev(HD_CURRENT_FORCE, omni_force);
    }
    hdEndFrame(driver->m_hHD);

    return HD_CALLBACK_CONTINUE;
}

//constructeur
GeomagicDriver::GeomagicDriver()
: d_deviceName(initData(&d_deviceName, std::string("Default Device"), "deviceName","Name of device Configuration"))
, d_positionBase(initData(&d_positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
, d_orientationBase(initData(&d_orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
, d_orientationTool(initData(&d_orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool"))
, d_dh_theta(initData(&d_dh_theta, Vec6d(0,0,0,M_PI/2,M_PI/2,M_PI/2), "dh_teta","Denavit theta"))
, d_dh_alpha(initData(&d_dh_alpha, Vec6d(0,-M_PI/2,0,M_PI/2,-M_PI/2,M_PI/2), "dh_alpha","Denavit alpha"))
, d_dh_d(initData(&d_dh_d, Vec6d(0,0,0,0,13.3333,0), "dh_d","Denavit d"))
, d_dh_a(initData(&d_dh_a, Vec6d(0,0,13.3333,0,0,0), "dh_a","Denavit a"))
, d_angle(initData(&d_angle, "angle","Angluar values of joint (rad)"))
, d_scale(initData(&d_scale, 1.0, "scale","Default scale applied to the Phantom Coordinates"))
, d_forceScale(initData(&d_forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
, d_omniVisu(initData(&d_omniVisu, false, "drawDeviceFrame", "Visualize the frame of the interface in the virtual scene"))
, d_posDevice(initData(&d_posDevice, "positionDevice", "position of the base of the part of the device"))
, d_button_1(initData(&d_button_1,"button1","Button state 1"))
, d_button_2(initData(&d_button_2,"button2","Button state 2"))
{
    this->f_listening.setValue(true);
    m_forceFeedback = NULL;
}

GeomagicDriver::~GeomagicDriver()
{
	hdMakeCurrentDevice(m_hHD);

	hdStopScheduler();

	for (std::vector< HDCallbackCode >::iterator i = m_hStateHandles.begin();
		i != m_hStateHandles.end(); ++i)
	{
		hdUnschedule(*i);
	}
	m_hStateHandles.clear();

	hdDisableDevice(m_hHD);
}

//executed once at the start of Sofa, initialization of all variables excepts haptics-related ones
void GeomagicDriver::init() {

	HDErrorInfo error;

	HDSchedulerHandle hStateHandle = HD_INVALID_HANDLE;
	m_hHD = hdInitDevice(d_deviceName.getValue().c_str());

	if (HD_DEVICE_ERROR(error = hdGetError())) {
		std::cerr << "[NewOmni] Failed to initialize the device called " << d_deviceName.getValue().c_str() << std::endl;
		return;
	}

	hdMakeCurrentDevice(m_hHD);
	hdEnable(HD_FORCE_OUTPUT);
	//    hdEnable(HD_MAX_FORCE_CLAMPING);

	hStateHandle = hdScheduleAsynchronous(stateCallback, this, HD_MAX_SCHEDULER_PRIORITY);
	m_hStateHandles.push_back(hStateHandle);

	hStateHandle = hdScheduleAsynchronous(copyDeviceDataCallback, this, HD_MIN_SCHEDULER_PRIORITY);
	m_hStateHandles.push_back(hStateHandle);

	if (HD_DEVICE_ERROR(error = hdGetError()))
	{
		printError(&error, "erreur avec le device Default PHANToM");
		return;
	}

    reinit();
}

void GeomagicDriver::bwdInit() {
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
    m_forceFeedback = context->get<ForceFeedback>(this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);

   
	hdStartScheduler();

	HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        std::cout<<"[NewOmni] Failed to start the scheduler"<<std::endl;
		if (m_hStateHandles.size()) m_hStateHandles[0] = HD_INVALID_HANDLE;
		return;
    }
    updatePosition();
}

Mat<4,4, GLdouble> GeomagicDriver::compute_dh_Matrix(double teta,double alpha, double a, double d) {
    Mat<4,4, GLdouble> M;

    const double dh_ct = cos(teta);
    const double dh_st = sin(teta);
    const double dh_ca = cos(alpha);
    const double dh_sa = sin(alpha);
    const double dh_a = a;
    const double dh_d = d;

    M[0][0] = dh_ct;
    M[0][1] = -dh_st*dh_ca;
    M[0][2] = dh_st*dh_sa;
    M[0][3] = dh_a*dh_ct;

    M[1][0] = dh_st;
    M[1][1] = dh_ct*dh_ca;
    M[1][2] = -dh_ct*dh_sa;
    M[1][3] = dh_a*dh_st;

    M[2][0] = 0;
    M[2][1] = dh_sa;
    M[2][2] = dh_ca;
    M[2][3] = dh_d;

    M[3][0] = 0;
    M[3][1] = 0;
    M[3][2] = 0;
    M[3][3] = 1;

    return M;
}


void GeomagicDriver::reinit() {
    Quat * q_b = d_orientationBase.beginEdit();
    q_b->normalize();
    d_orientationBase.endEdit();

    Quat * q_t = d_orientationTool.beginEdit();
    q_t->normalize();
    d_orientationTool.endEdit();

    for (int i=0;i<NBJOINT;i++) m_dh_matrices[i] = compute_dh_Matrix(d_dh_theta.getValue()[i],d_dh_alpha.getValue()[i],d_dh_a.getValue()[i],d_dh_d.getValue()[i]);
}

void GeomagicDriver::updatePosition() {

    Mat3x3d mrot;

    Vector6 & angle = *d_angle.beginEdit();
    GeomagicDriver::Coord & posDevice = *d_posDevice.beginEdit();
    bool & button_1 = *d_button_1.beginEdit();
    bool & button_2 = *d_button_2.beginEdit();

    const Vector3 & positionBase = d_positionBase.getValue();
    const Quat & orientationBase = d_orientationBase.getValue();
    const Quat & orientationTool = d_orientationTool.getValue();
    const double & scale = d_scale.getValue();

    //copy button state
    button_1 = m_simuData.buttonState & HD_DEVICE_BUTTON_1;
    button_2 = m_simuData.buttonState & HD_DEVICE_BUTTON_2;

    //copy angle
    angle[0] =  m_simuData.angle1[0];
    angle[1] =  m_simuData.angle1[1];
    angle[2] =  -(M_PI/2)+m_simuData.angle1[2]-m_simuData.angle1[1];
    angle[3] =  -(M_PI/2)-m_simuData.angle2[0];
    angle[4] = m_simuData.angle2[1];
    angle[5] =  -(M_PI/2)-m_simuData.angle2[2];

    //copy the position of the tool
    Vector3 position;
    position[0] = m_simuData.transform[12+0] * 0.1;
    position[1] = m_simuData.transform[12+1] * 0.1;
    position[2] = m_simuData.transform[12+2] * 0.1;

    //copy rotation of the tool
    Quat orientation;
    for (int u=0; u<3; u++)
        for (int j=0; j<3; j++)
            mrot[u][j] = m_simuData.transform[j*4+u];
    orientation.fromMatrix(mrot);

    //compute the position of the tool (according to positionbase, orientation base and the scale
    posDevice.getCenter() = positionBase + orientationBase.rotate(position*scale);
    posDevice.getOrientation() = orientationBase * orientation * orientationTool;

    d_posDevice.endEdit();
    d_angle.endEdit();
    d_button_1.endEdit();
    d_button_2.endEdit();
}

void GeomagicDriver::getMatrix(Mat<4,4, GLdouble> & M1, int index, double teta) {
    const double ct = cos(teta);
    const double st = sin(teta);

    const Mat<4,4, GLdouble> & M = m_dh_matrices[index];

    M1[0][0] =  M[0][0]*ct + M[0][1]*st;
    M1[0][1] = -M[0][0]*st + M[0][1]*ct;
    M1[0][2] =  M[0][2];
    M1[0][3] =  M[0][3];

    M1[1][0] =  M[1][0]*ct + M[1][1]*st;
    M1[1][1] = -M[1][0]*st + M[1][1]*ct;
    M1[1][2] =  M[1][2];
    M1[1][3] =  M[1][3];

    M1[2][0] =  M[2][0]*ct + M[2][1]*st;
    M1[2][1] = -M[2][0]*st + M[2][1]*ct;
    M1[2][2] =  M[2][2];
    M1[2][3] =  M[2][3];

    M1[3][0] =  M[3][0]*ct + M[3][1]*st;
    M1[3][1] = -M[3][0]*st + M[3][1]*ct;
    M1[3][2] =  M[3][2];
    M1[3][3] =  M[3][3];
}

void GeomagicDriver::draw(const sofa::core::visual::VisualParams* vparams) {
#ifndef SOFA_NO_OPENGL
    if (!d_omniVisu.getValue()) return;

    Mat<4,4, GLdouble> M;

    glDisable(GL_LIGHTING);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    //Replace the base of the omni
    sofa::defaulttype::SolidTypes<double>::Transform transform(d_positionBase.getValue(),d_orientationBase.getValue());
    transform.writeOpenGlMatrix((double*) M.ptr());
    glMultMatrixd((double*) M.ptr());

    //scale the omni
    glScaled(d_scale.getValue(),d_scale.getValue(),d_scale.getValue());

    //change the initial frame of the omni
    glMultMatrixd((double*) initialTransform.ptr());

    glPopMatrix();

    vparams->drawTool()->drawArrow(d_posDevice.getValue().getCenter(), d_posDevice.getValue().getCenter() + d_posDevice.getValue().getOrientation().rotate(Vector3(2,0,0)*d_scale.getValue()), d_scale.getValue()*0.1, Vec4f(1,0,0,1) );
    vparams->drawTool()->drawArrow(d_posDevice.getValue().getCenter(), d_posDevice.getValue().getCenter() + d_posDevice.getValue().getOrientation().rotate(Vector3(0,2,0)*d_scale.getValue()), d_scale.getValue()*0.1, Vec4f(0,1,0,1) );
    vparams->drawTool()->drawArrow(d_posDevice.getValue().getCenter(), d_posDevice.getValue().getCenter() + d_posDevice.getValue().getOrientation().rotate(Vector3(0,0,2)*d_scale.getValue()), d_scale.getValue()*0.1, Vec4f(0,0,1,1) );
#endif
}

void GeomagicDriver::computeBBox(const core::ExecParams*  params, bool  ) {
    SReal minBBox[3] = {1e10,1e10,1e10};
    SReal maxBBox[3] = {-1e10,-1e10,-1e10};

    minBBox[0] = d_posDevice.getValue().getCenter()[0]-d_positionBase.getValue()[0]*d_scale.getValue();
    minBBox[1] = d_posDevice.getValue().getCenter()[1]-d_positionBase.getValue()[1]*d_scale.getValue();
    minBBox[2] = d_posDevice.getValue().getCenter()[2]-d_positionBase.getValue()[2]*d_scale.getValue();

    maxBBox[0] = d_posDevice.getValue().getCenter()[0]+d_positionBase.getValue()[0]*d_scale.getValue();
    maxBBox[1] = d_posDevice.getValue().getCenter()[1]+d_positionBase.getValue()[1]*d_scale.getValue();
    maxBBox[2] = d_posDevice.getValue().getCenter()[2]+d_positionBase.getValue()[2]*d_scale.getValue();

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
}

void GeomagicDriver::handleEvent(core::objectmodel::Event *event) {
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event)) {
		if (m_hStateHandles.size() && m_hStateHandles[0] == HD_INVALID_HANDLE) return;
		updatePosition();
    }
}

int GeomagicDriverClass = core::RegisterObject("Driver allowing interfacing with Geomagic haptic devices.")
.add< GeomagicDriver >()
.addAlias("DefaultHapticsDevice")
;

SOFA_DECL_CLASS(GeomagicDriver)

} // namespace controller

} // namespace component

} // namespace sofa
