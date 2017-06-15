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

#include <sofa/core/behavior/BaseController.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaUserInteraction/Controller.h>
//#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/visual/VisualParams.h>
//#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <iostream>
#include <sstream>
#include <MyListener.h>

namespace sofa
{
namespace simulation { class Node; }

namespace component
{

namespace controller
{

enum finger_name 
    {
        THUMB = 0,
        FOREFINGER,
        MIDDLE_FINGER,
        RING_FINGER,
        LITTLE_FINGER,
        ALL_FINGERS
    } ;

enum gestureTypeEnum
    {
        TYPE_SWIPE = 0,
        TYPE_CIRCLE,
        TYPE_SCREEN_TAP,
        TYPE_KEY_TAP,
        TYPE_PINCH,
        TYPE_V_SIGN_AND_2ND_HAND,
        TYPES_COUNT,
        GRASP_AND_RELEASE
    };


class LeapMotionDriver : public sofa::component::controller::Controller
{

public:
    SOFA_CLASS(LeapMotionDriver, sofa::component::controller::Controller);	

    int animEventCounter;

    Data< double > scale;
    Data< Vec3d > translation;
    Data< sofa::defaulttype::Vector3 > rotation;
    Data< Rigid3dTypes::Coord > handPalmCoordinate;
    Data< Vec3d > sphereCenter;
    Data< double > sphereRadius;
    Data< sofa::helper::vector< Rigid3dTypes::Coord > > fingersCoordinates;
    Data< int > gestureType;
    Data< Vec3d > gesturePosition;
    Data< Vec3d > gestureDirection;
    Data< int > scrollDirection;
    Data< bool > displayHand;
    Data< double > speed;

    LeapMotionDriver();
    virtual ~LeapMotionDriver();

    void init();
    void bwdInit();
    void reset();
    void reinit();

    void cleanup();

    Vec3d verticeOnRadius(Vec3d center, Vec3d vecOnCirclePlane, Vec3d orthoVecOnCirclePlane, double radius, double radAngle);
    void computeFingerJoints(int i, sofa::helper::WriteAccessor<Data<sofa::helper::vector<RigidCoord<3,double> > > >* fingersCoordsArray);
    void draw(const sofa::core::visual::VisualParams* vparams);
    void computeBBox(const sofa::core::ExecParams *, bool);
    void applyRotation (Rigid3dTypes::Coord* rigidToRotate);
    void reinitVisual();
    void TimeAfterGesture(double,Data< double >);

private:
	
    static ::Leap::Controller* getLeapController()
    {
        static ::Leap::Controller leapController;
        return &leapController;
    }

    void handleEvent(sofa::core::objectmodel::Event *);

    Leap::Config leapConfig;
    MyListener myListener;

    bool drawInteractionBox;
    bool leapConnected;
    sofa::helper::system::thread::CTime savedTime;
    double slideDisplayDuration;
    int64_t currentFrame;
    bool isNewFrame;
    std::vector<int> tmpIds;

    int lastGestureType;
    int64_t lastGestureFrameId;
    std::vector< std::pair<int, int64_t> > lastGesturesFrameIds;
    bool pinchGestureRecognized, VSignGestureRecognized, secondHandRecognized, VSignAndSecondHandGestureRecognized, GraspAndReleaseGestureRecognized;

};

} // namespace controller

} // namespace component

} // namespace sofa
