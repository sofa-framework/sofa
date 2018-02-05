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

#include "LeapMotionDriver.h"
#ifdef SOFA_HAVE_BOOST
#include <boost/thread/thread.hpp>
#endif
#include <SofaGeneralVisual/VisualTransform.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>


namespace sofa
{

namespace component
{

namespace controller
{

#define SCROLL_DIRECTION_THRESHOLD 0.5
#define SWIPE_MIN_VELOCITY 500
#define SWIPE_MAX_VELOCITY 6000
#define PINCH_DISTANCE_THRESHOLD 40 // a pinch will be detected if the distance between two fingers is smaller than this thresold
#define PINCH_DOTPRODUCT_THRESHOLD 0.8
int32_t fingersIdsArray[5] = {-1,-1,-1,-1,-1};

LeapMotionDriver::LeapMotionDriver()
    : scale(initData(&scale, 1.0, "scale","Default scale applied to the Leap Motion Coordinates. "))
    , translation(initData(&translation, Vec3d(0,0,0), "translation","Position of the tool/hand in the Leap Motion reference frame"))
    , rotation(initData(&rotation, sofa::defaulttype::Vector3(), "rotation", "Rotation of the DOFs of the hand"))
    , handPalmCoordinate(initData(&handPalmCoordinate, "handPalmCoordinate","Coordinate of the hand detected by the Leap Motion"))
    , sphereCenter(initData(&sphereCenter, "sphereCenter","Center of the sphere of the hand detected by the Leap Motion"))
    , sphereRadius(initData(&sphereRadius, "sphereRadius","Radius of the sphere of the hand detected by the Leap Motion"))
    , fingersCoordinates(initData(&fingersCoordinates, sofa::helper::vector<Rigid3dTypes::Coord>(1,Rigid3dTypes::Coord(sofa::defaulttype::Vector3(0,0,0),Quat(0,0,0,1))), "fingersCoordinates","Coordinate of the fingers detected by the Leap Motion"))
    , gestureType(initData(&gestureType, int(-1) ,"gestureType","Type of the current gesture detected by the Leap Motion"))
    , gesturePosition(initData(&gesturePosition, "gesturePosition","Position of the current gesture detected by the Leap Motion"))
    , gestureDirection(initData(&gestureDirection, "gestureDirection","Direction of the current gesture detected by the Leap Motion"))
    , scrollDirection(initData (&scrollDirection, int(0), "scrollDirection", "Enter 0 if no scrolling (1 if scoll increases the value, 2 if scroll decreases it)"))
    , displayHand(initData (&displayHand, false, "displayHand", "display the hand detected by the Leap Motion"))
{
    this->f_listening.setValue(true);
    slideDisplayDuration = sofa::helper::system::thread::CTime::getTime();
}


LeapMotionDriver::~LeapMotionDriver() {
    // Remove the sample listener when done
    if (leapConnected)
    {
        getLeapController()->removeListener(myListener);
    }
}


void LeapMotionDriver::cleanup()
{
    sout << "LeapMotionDriver::cleanup()" << sendl;
}


void LeapMotionDriver::init()
{	
	bool leapEnabled = true;
	
	leapConnected = false;
        if (leapEnabled)
        {
            std::cout << "[LeapMotion] Connecting";
            for (unsigned int n=0; n<15 && !leapConnected; n++)
            {
                std::cout << ".";
                sofa::helper::system::thread::CTime::sleep(0.1);
                leapConnected = getLeapController()->isConnected();
            }
            std::cout << "." << std::endl;
	}

        if (leapConnected)
        {
            leapConfig = getLeapController()->config();
            leapConfig.setFloat("Gesture.ScreenTap.MinForwardVelocity",50);
            leapConfig.setFloat("Gesture.ScreenTap.HistorySeconds",0.1);
            leapConfig.setFloat("Gesture.ScreenTap.MinDistance",3.0);

            leapConfig.setFloat("Gesture.KeyTap.MinDownVelocity",10);
            leapConfig.setFloat("Gesture.KeyTap.HistorySeconds",0.1);
            leapConfig.setFloat("Gesture.KeyTap.MinDistance",5.0);

            leapConfig.setFloat("Gesture.Swipe.MinLength",120);
            leapConfig.setFloat("Gesture.Swipe.MinVelocity",SWIPE_MIN_VELOCITY);

            leapConfig.setFloat("Gesture.Circle.MinArc", 2*M_PI);
            leapConfig.setFloat("Gesture.Circle.MinRadius", 25.0);
            leapConfig.save();
            getLeapController()->addListener(myListener);
            drawInteractionBox = false;
        }
        else if (leapEnabled)
		serr << "Device not detected" << sendl;


	//initialisation of the fingers coordinates array
        helper::WriteAccessor<Data<sofa::helper::vector<RigidCoord<3,double> > > > fingerCoords = fingersCoordinates;
        for (int i=0; i<14; i++)
            fingerCoords.push_back(fingerCoords[i]);
	
	scrollDirection.setValue(0);

        lastGestureType = TYPE_SWIPE;
        for (int i=0; i<TYPES_COUNT; i++)
        {
            lastGesturesFrameIds.push_back(std::make_pair(i,0));
	}
	lastGestureFrameId = 0;
}


void LeapMotionDriver::bwdInit()
{
    sout<<"LeapMotionDriver::bwdInit()"<<sendl;
}


void LeapMotionDriver::reset()
{
    sout<<"LeapMotionDriver::reset()" << sendl;
    scrollDirection.setValue(0);
    this->reinit();
}


void LeapMotionDriver::reinitVisual() {}


void LeapMotionDriver::reinit()
{
    this->bwdInit();
    this->reinitVisual();
}


Vec3d LeapMotionDriver::verticeOnRadius(Vec3d center, Vec3d vecOnCirclePlane, Vec3d orthoVecOnCirclePlane, double radius, double radAngle)
{
	vecOnCirclePlane.normalize();
	orthoVecOnCirclePlane.normalize();
	
	return Vec3d(center + radius*(vecOnCirclePlane*cos(radAngle) + orthoVecOnCirclePlane*sin(radAngle)));
}

void LeapMotionDriver::computeFingerJoints(int i, helper::WriteAccessor<Data<sofa::helper::vector<RigidCoord<3,double> > > >* fingersCoordsArray )
{
	//palm normal
	Mat3x3d palmMatrix;
	handPalmCoordinate.getValue().writeRotationMatrix(palmMatrix);
	Vec3d palmDirection = -Vec3d(palmMatrix.col(0));
	palmDirection.normalize();
	Vec3d palmNormal = -Vec3d(palmMatrix.col(2));
	palmNormal.normalize();
		
	//\\Projections\\//
	Vec3d v = fingersCoordinates.getValue()[3*i].getCenter() - handPalmCoordinate.getValue().getCenter();
	//fingerTip projection into vertical palm plane
	double dist = dot(v,palmDirection);
        //Vec3d v_proj_point = fingersCoordinates.getValue()[3*i].getCenter() - dist*palmDirection;

	//fingerTip projection into horizontal palm plane
	dist = dot(v,palmNormal);
	Vec3d h_proj_point = fingersCoordinates.getValue()[3*i].getCenter() - dist*palmNormal;
	
		
	//metacarpophalangeal joint
	Vec3d palmToTipProjectionDirection = handPalmCoordinate.getValue().getCenter() - h_proj_point;
	palmToTipProjectionDirection.normalize();
	Quat McpQuat = Quat().createQuaterFromFrame(palmNormal, palmToTipProjectionDirection.cross(palmNormal),palmToTipProjectionDirection);
	Rigid3dTypes::Coord Mcp = Rigid3dTypes::Coord(handPalmCoordinate.getValue().getCenter() - palmToTipProjectionDirection.mulscalar(30*scale.getValue()), McpQuat);
	(*fingersCoordsArray)[3*i+1] = Mcp;

	//proximal interphalangial joint
	double pipAngleCorrection = (h_proj_point - fingersCoordinates.getValue()[3*i].getCenter()).norm()/scale.getValue()/100.0;
	if(dist > 0) {pipAngleCorrection = -2*pipAngleCorrection;}
	Vec3d pipPos = verticeOnRadius(Mcp.getCenter(),Vec3d(palmMatrix.col(2)),(handPalmCoordinate.getValue().getCenter()-Mcp.getCenter()),(Mcp.getCenter()-fingersCoordinates.getValue()[3*i].getCenter()).norm()/2.2 ,1.5*M_PI+pipAngleCorrection);
	(*fingersCoordsArray)[3*i+2] = Rigid3dTypes::Coord(pipPos, McpQuat);
}


void LeapMotionDriver::draw(const sofa::core::visual::VisualParams* vparams)
{
	if (!vparams->displayFlags().getShowVisualModels()) return;

	if(displayHand.getValue()) {
            helper::gl::GlText text;
            //text.setText("SLICE");
            //text.update(Vector3(myListener.getStabilizedPalmPosition().toFloatPointer()).mulscalar(scale.getValue()));
            //text.update(0.0005);
	    text.draw();
	
            if(leapConnected && drawInteractionBox )
            {
                Vec3d interactionBoxParallelepipedVecs[5];
                Vec3d leapInteractionBoxCenter = Vec3d(myListener.getInteractionBox().center().toFloatPointer());
                double halfWidth = myListener.getInteractionBox().width()/2.0;
                double halfHeight = myListener.getInteractionBox().height()/2.0;
                double halfDepth = myListener.getInteractionBox().depth()/2.0;
                interactionBoxParallelepipedVecs[0] = Vec3d(leapInteractionBoxCenter.x()-halfWidth, leapInteractionBoxCenter.y() - halfHeight, leapInteractionBoxCenter.z() - halfDepth);
                interactionBoxParallelepipedVecs[1] = Vec3d(leapInteractionBoxCenter.x()-halfWidth, leapInteractionBoxCenter.y() - halfHeight, leapInteractionBoxCenter.z() + halfDepth);
                interactionBoxParallelepipedVecs[2] = Vec3d(leapInteractionBoxCenter.x()-halfWidth, leapInteractionBoxCenter.y() + halfHeight, leapInteractionBoxCenter.z() + halfDepth);
                interactionBoxParallelepipedVecs[3] = Vec3d(leapInteractionBoxCenter.x()-halfWidth, leapInteractionBoxCenter.y() + halfHeight, leapInteractionBoxCenter.z() - halfDepth);
                interactionBoxParallelepipedVecs[4] = Vec3d(2.0*halfWidth, 0.0, 0.0);

                Quat q = Quat().createQuaterFromEuler(rotation.getValue().mulscalar(M_PI / 180.0));
                for(int i=0; i<5; i++)
                {
                        interactionBoxParallelepipedVecs[i] = q.rotate(interactionBoxParallelepipedVecs[i].mulscalar(scale.getValue()));
                        interactionBoxParallelepipedVecs[i] += translation.getValue();
                }
///FIXME: draw
//		helper::gl::drawEmptyParallelepiped(interactionBoxParallelepipedVecs[0],interactionBoxParallelepipedVecs[1],interactionBoxParallelepipedVecs[2],interactionBoxParallelepipedVecs[3],interactionBoxParallelepipedVecs[4],1.25*scale.getValue());
            }

            if (leapConnected)
            {
                float coordMatrix[16];

                //draw palm as a torus
                handPalmCoordinate.getValue().writeOpenGlMatrix(coordMatrix);
                helper::gl::drawTorus( coordMatrix, 2.5*scale.getValue(), 28*scale.getValue(), 30, Vec3i(255,215,180) );
                //helper::gl::drawTorus( coordMatrix, 2.5*scale.getValue(), 23*scale.getValue(), 30, Vec3i(255,215,180) );
                //helper::gl::drawTorus( coordMatrix, 4*scale.getValue(), 16.5*scale.getValue(), 30, Vec3i(255,215,180) );
                //helper::gl::drawTorus( coordMatrix, 4*scale.getValue(), 8.5*scale.getValue(), 30, Vec3i(255,215,180) );


                //draw palm Normal as a cylinder
                //Mat3x3d palmMatrix;
                //handPalmCoordinate.getValue().writeRotationMatrix(palmMatrix);
                //Vec3d palmNormal = -Vec3d(palmMatrix.col(2));
                //palmNormal.normalize();
                //helper::gl::drawCylinder(handPalmCoordinate.getValue().getCenter(), handPalmCoordinate.getValue().getCenter() + palmNormal.mulscalar(3.0) ,0.25,8);

                for(int i=0; i<5; i++)
                {
                    //draw fingerTip as a torus
                    fingersCoordinates.getValue()[3*i].writeOpenGlMatrix(coordMatrix);
                    helper::gl::drawTorus(coordMatrix,0.9*scale.getValue(),4.0*scale.getValue(),30, Vec3i(255-(i*55)%255, 255-((5-i)*75)%255, (i*60)%255));

                    //draw metacarpus as a cylinder
                    helper::gl::drawCylinder(handPalmCoordinate.getValue().getCenter(),fingersCoordinates.getValue()[3*i+1].getCenter() ,2.25*scale.getValue(),8);

                    //draw metacarpophalangeal joint as a torus
                    fingersCoordinates.getValue()[3*i+1].writeOpenGlMatrix(coordMatrix);
                    helper::gl::drawTorus(coordMatrix,scale.getValue(),5.0*scale.getValue(),30,Vec3i(255,215,180));

                    //draw proximal interphalangial joint
                    helper::gl::drawCylinder(fingersCoordinates.getValue()[3*i+1].getCenter(),fingersCoordinates.getValue()[3*i+2].getCenter() ,2.25*scale.getValue(),8);
                    helper::gl::drawCylinder(fingersCoordinates.getValue()[3*i+2].getCenter(), fingersCoordinates.getValue()[3*i].getCenter(),2.25*scale.getValue(),8);
                    helper::gl::drawSphere(fingersCoordinates.getValue()[3*i+2].getCenter(),2.0*scale.getValue(),15,15);

                    //draw leap sphere of the hand
                    //helper::gl::drawSphere(sphereCenter.getValue(),0.1*sphereRadius.getValue()*scale.getValue(),15,15);
                }
            }
	}
}


void LeapMotionDriver::computeBBox(const core::ExecParams * params, bool /*onlyVisible=false*/ )
{
    const double max_real = std::numeric_limits<double>::max();
    const double min_real = -std::numeric_limits<double>::max();
    double maxBBox[3] = {min_real,min_real,min_real};
    double minBBox[3] = {max_real,max_real,max_real};

    for(int i=0; i<15; i++)
    {
        for( int c=0; c<3; c++)
        {
            minBBox[c] = (fingersCoordinates.getValue()[i][c] < minBBox[c]) ? fingersCoordinates.getValue()[i][c] : minBBox[c] ;
            maxBBox[c] = (fingersCoordinates.getValue()[i][c] > maxBBox[c]) ? fingersCoordinates.getValue()[i][c] : maxBBox[c] ;
        }
    }

    Mat3x3d matrix;
    handPalmCoordinate.getValue().writeRotationMatrix(matrix);
    Vec3d toFrontHandPalmDirection = Vec3d(matrix.col(0));
    double palmDiag = toFrontHandPalmDirection.mulscalar(20.0*scale.getValue()).norm() * sqrt(2.0);

    for( int c=0; c<3; c++)
    {
        minBBox[c] = ((handPalmCoordinate.getValue()[c] - palmDiag) < minBBox[c]) ? (handPalmCoordinate.getValue()[c] - palmDiag) : minBBox[c] ;
        maxBBox[c] = ((handPalmCoordinate.getValue()[c] + palmDiag) > maxBBox[c]) ? (handPalmCoordinate.getValue()[c] + palmDiag) : maxBBox[c] ;
    }
    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<double>(minBBox,maxBBox));
}


void LeapMotionDriver::applyRotation (Rigid3dTypes::Coord* rigidToRotate)
{
    Quat q = Quat().createQuaterFromEuler(rotation.getValue().mulscalar(M_PI / 180.0));
    (*rigidToRotate).getCenter() = q.rotate((*rigidToRotate).getCenter());
    (*rigidToRotate).getOrientation() = q * (*rigidToRotate).getOrientation();
}


void LeapMotionDriver::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        if (ev->getKey() == 'I' || ev->getKey() == 'i')
        {
            drawInteractionBox = !drawInteractionBox;
        }

        /*Does it work ?*/
        else if (ev->getKey() == 'Q' || ev->getKey() == 'q')
        {
            gestureType.setValue(TYPE_V_SIGN_AND_2ND_HAND);
            gesturePosition.setValue(Vec3d(currentFrame,0,0));
            gestureDirection.setValue(Vec3d(0,0,0));

            lastGestureType = TYPE_V_SIGN_AND_2ND_HAND;
            lastGestureFrameId = currentFrame;
            lastGesturesFrameIds.at(TYPE_V_SIGN_AND_2ND_HAND).second = currentFrame;
        }
    }


    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {

//----- Get hand palm coordinates
        if (leapConnected )
        {
            Vector3 handPalmPosition = Vector3(myListener.getStabilizedPalmPosition().toFloatPointer());
            Quat handPalmRotation = myListener.getPalmRotation();

            Rigid3dTypes::Coord tmpHandPalmCoord = Rigid3dTypes::Coord(scale.getValue()*handPalmPosition, handPalmRotation);
            applyRotation(&tmpHandPalmCoord);
            tmpHandPalmCoord.getCenter() += translation.getValue();
            handPalmCoordinate.setValue(tmpHandPalmCoord);
        }

//----- Get hand sphere
        if (leapConnected )
        {
            Vector3 handSphereCenter = Vector3(myListener.getHandSphereCenter().toFloatPointer());
            float handSphereRadius = myListener.getHandSphereRadius();
            Quat q = Quat().createQuaterFromEuler(rotation.getValue().mulscalar(M_PI / 180.0));
            Vec3d tmpHandSphereCenter = q.rotate(scale.getValue()*handSphereCenter);
            tmpHandSphereCenter += translation.getValue();
            sphereCenter.setValue(tmpHandSphereCenter);
            sphereRadius.setValue(scale.getValue()*handSphereRadius);
        }


//----- Get fingers coordinates

        int fingerCount = std::min(myListener.getFingers().count(),5);

        //----begin finger registration init
        int registeredFingersCount = 0;
        int nbFrame=0;
        for(int i=0; i<5; i++)
        {
            if(fingersIdsArray[i]!=-1)
                registeredFingersCount++;
        }

        int32_t newFingersIdsArray[5] = {-1,-1,-1,-1,-1};
        tmpIds.clear();
        //----end finger registration init

        helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > fingerCoords = fingersCoordinates;
        for(int i=0; i<15; i++)
        {
            fingerCoords[i] = handPalmCoordinate.getValue();
        }

        for(int i=0; i<fingerCount; i++)
        {
            nbFrame++;

            Vec3d xAxis = Vec3d(myListener.getFingers()[i].direction().toFloatPointer());
            xAxis.normalize();
            Vector3 yAxis;
            Vector3 zAxis(1.0, 0.0, 0.0);
            if ( fabs(dot(zAxis, xAxis)) > 0.7)
                zAxis = Vector3(0.0, 0.0, 1.0);

            yAxis = zAxis.cross(xAxis);
            yAxis.normalize();
            zAxis = xAxis.cross(yAxis);
            zAxis.normalize();

            Vec3d fingerPosition = Vec3d(myListener.getFingers()[i].stabilizedTipPosition().toFloatPointer());
            Quat fingerOrientation = Quat().createQuaterFromFrame(yAxis, zAxis, xAxis);

            Rigid3dTypes::Coord tmpFingerCoord = Rigid3dTypes::Coord(scale.getValue()*fingerPosition, fingerOrientation);
            applyRotation(&tmpFingerCoord);
            tmpFingerCoord.getCenter() += translation.getValue();
            fingerCoords[3*i] = tmpFingerCoord;
            computeFingerJoints(i, &fingerCoords);

            bool alreadyRegistered=false;
            int32_t fingerId;
            fingerId = myListener.getFingers()[i].id();


            //----begin finger registration
            int fIdIter;
            for ( fIdIter = 0; fIdIter < 5; fIdIter++)
            {
                if(fingersIdsArray[fIdIter]==fingerId)
                {
                    newFingersIdsArray[fIdIter] = fingerId;
                    alreadyRegistered = true;
                }
            }

            fIdIter = 0;

            if(!alreadyRegistered)
                tmpIds.insert(tmpIds.end(),fingerId);

        }//foreach finger

        for(size_t j=0; j<tmpIds.size(); j++)
        {
            int fIdIter=0;
            while(fIdIter<5 && newFingersIdsArray[fIdIter]!=-1 )
            {
                fIdIter++;
            }

            if(fIdIter<5)
                newFingersIdsArray[fIdIter] = tmpIds[j];
            else
                std::cout << "ID insersion failed, array is full" << std::endl;
        }

        for (int k=0; k<5; k++)
            fingersIdsArray[k] = newFingersIdsArray[k];

        //----end finger registration





//--------------------------------  OUR Handle Gestures ----------------------------

        // -------------------------- PINCH Gesture ----------------------------
        Mat3x3d fingerMatrix;
        pinchGestureRecognized = false;
        double pinchDotproduct;
        Vec3d pinchPosition;

        for(int i=0; i<fingerCount; i++)
        {
            fingerCoords[3*i].getOrientation().toMatrix(fingerMatrix);
            Vec3d finger1Dir = Vec3d(fingerMatrix.col(0));
            double distance;
            for(int fIter= (i+1); fIter<fingerCount; fIter++)
            {
                distance = (fingerCoords[3*i].getCenter()-fingerCoords[3*fIter].getCenter()).norm() / scale.getValue();
                if(distance < PINCH_DISTANCE_THRESHOLD)
                {
                    fingerCoords[3*fIter].getOrientation().toMatrix(fingerMatrix);
                    Vec3d finger2Dir = Vec3d(fingerMatrix.col(0));
                    pinchDotproduct = dot(finger1Dir,finger2Dir);

                    if(pinchDotproduct < PINCH_DOTPRODUCT_THRESHOLD)
                    {
                        if(!pinchGestureRecognized)
                        {
                            pinchPosition = fingerCoords[3*i].getCenter() + (fingerCoords[3*fIter].getCenter()-fingerCoords[3*i].getCenter()).divscalar(2.0);
                        }
                        pinchGestureRecognized = true;
                    }
                }
            }
        }
        if(pinchGestureRecognized)
        {
            if(((currentFrame-lastGesturesFrameIds.at(TYPE_PINCH).second)/myListener.getCurrentFramesPerSecond())<5.0)
            {
                if(f_printLog.getValue())
                    std::cout << "Pinch rejected" << std::endl;
            }
            else
            {
                if(f_printLog.getValue())
                    std::cout << "PINCH gesture detected"<<std::endl;
                gestureType.setValue(TYPE_PINCH);
                gesturePosition.setValue(pinchPosition);
                gestureDirection.setValue(Vec3d(0,0,0));

                lastGestureType = TYPE_PINCH;
                lastGestureFrameId = currentFrame;
                lastGesturesFrameIds.at(TYPE_PINCH).second = currentFrame;
            }
        }

        // -------------------------- TYPE_V_SIGN_ Gesture ----------------------------
        static int lastFingerCount = 0, beginVSignFrame = 0, lastSecondHandFingerCount = 0, beginSecondHandFrame = 0;
        double distance;

        //V SIGN
        if(fingerCount == 2)
        {
            if(lastFingerCount!=2)
            {
                beginVSignFrame = currentFrame;
            }
            else
            {
                distance=sqrt(pow(((fingersCoordinates.getValue(0)[0][0])-(fingersCoordinates.getValue(0)[1][0])),2)+pow(((fingersCoordinates.getValue(0)[0][1])-(fingersCoordinates.getValue(0)[1][1])),2)+pow(((fingersCoordinates.getValue(0)[0][2])-(fingersCoordinates.getValue(0)[1][2])),2));

                if((currentFrame-beginVSignFrame)/myListener.getFPS()>3.0 && distance<=160 && distance>=70)
                {
                    VSignGestureRecognized = true;
                    if(f_printLog.getValue())
                        std::cout << "V_SIGN gesture detected"<<std::endl;
                }
            }
        }
        else
        {
            VSignGestureRecognized = false;
        }

        lastFingerCount = fingerCount;


        //2ND HAND
        if(myListener.getSecondHand().isValid() && myListener.getSecondHandFingers().count()>=4)
        {
            if(lastSecondHandFingerCount<4)
            {
                beginSecondHandFrame = currentFrame;
            }
            else
            {
                if((currentFrame-beginSecondHandFrame)/myListener.getCurrentFramesPerSecond()>2.0)
                {
                    secondHandRecognized = true;
                    if(f_printLog.getValue())
                        std::cout << "Second hand detected"<<std::endl;
                }
            }
        }
        else
        {
            secondHandRecognized =  false;
        }

        lastSecondHandFingerCount = myListener.getSecondHandFingers().count();


        //------------------------------------------- GRASP_AND_RELEASE_GESTURE-----------------------------------------------------
        if (fingerCount==0 && lastFingerCount>=4 && myListener.getHand().palmPosition().isValid())
        {
            GraspAndReleaseGestureRecognized=true;

            if (GraspAndReleaseGestureRecognized && ((currentFrame-lastGesturesFrameIds.at(GRASP_AND_RELEASE).second)/myListener.getFPS())>2.0)
            {
                if(f_printLog.getValue())
                    std::cout << "GRASP gesture detected"<<std::endl;
                gestureType.setValue(GRASP_AND_RELEASE);
                gesturePosition.setValue(handPalmCoordinate.getValue().getCenter());
                gestureDirection.setValue(Vec3d(0,0,0));
            }

            lastGestureType = GRASP_AND_RELEASE;
            lastGestureFrameId = currentFrame;
            lastGesturesFrameIds.at(GRASP_AND_RELEASE).second = currentFrame;
        }


        if (!myListener.getGestures().isEmpty())
        {
            int nbGesturesInFrame = 0;
            for (int g = 0; g < myListener.getGestures().count(); ++g)
            {
                Gesture gesture = myListener.getGestures()[g];
                if ( gesture.isValid() && gesture.state()==Gesture::STATE_STOP)
                {
                    nbGesturesInFrame++;
                    switch (gesture.type())
                    {
                        // -------------------------- CIRCLE Gesture ----------------------------
                        case Gesture::TYPE_CIRCLE: {
                            if(((currentFrame-lastGesturesFrameIds.at(TYPE_CIRCLE).second)/myListener.getCurrentFramesPerSecond())<2.0)
                            {
                                if(f_printLog.getValue())
                                    std::cout << "CIRCLE REJECTED !" << std::endl;
                                break;
                            }

                            CircleGesture circle = gesture;
                            gestureType.setValue(TYPE_CIRCLE);
                            gesturePosition.setValue(Vec3d(circle.center().toFloatPointer()));
                            gestureDirection.setValue(Vec3d(0,0,0));
                            scrollDirection.setValue(0);

                            if(f_printLog.getValue())
                                std::cout << "CIRCLE gesture detected"<<std::endl;

                            lastGestureType = TYPE_CIRCLE;
                            lastGesturesFrameIds.at(TYPE_CIRCLE).second = currentFrame;
                            break;
                        }

                        // -------------------------- SWIPE Gesture ----------------------------
                        case Gesture::TYPE_SWIPE:
                        {
                            if(((currentFrame-lastGesturesFrameIds.at(TYPE_SWIPE).second)/myListener.getCurrentFramesPerSecond())<1.0 || ((currentFrame-lastGesturesFrameIds.at(TYPE_CIRCLE).second)/myListener.getCurrentFramesPerSecond())<1.0 )
                            {
                                if(f_printLog.getValue())
                                    std::cout << "SWIPE REJECTED !" << std::endl;
                                break;
                            }

                            SwipeGesture swipe=gesture;
                            if(swipe.speed() < SWIPE_MIN_VELOCITY)
                                break;
                            gestureType.setValue(TYPE_SWIPE);
                            gesturePosition.setValue(Vec3d(swipe.position().toFloatPointer()));
                            gestureDirection.setValue(Vec3d(swipe.direction().toFloatPointer()));

                            if(f_printLog.getValue())
                                std::cout << "SWIPE gesture detected"<<std::endl;

                            double leapScrollDirection = gestureDirection.getValue()[0];
                            leapScrollDirection /= gestureDirection.getValue().norm();

                            if (leapScrollDirection < -SCROLL_DIRECTION_THRESHOLD)
                            {
                                scrollDirection.setValue(1);
                            }
                            else if (leapScrollDirection > SCROLL_DIRECTION_THRESHOLD)
                            {
                                scrollDirection.setValue(2);
                            }
                            else
                            {
                                break;
                            }

                            lastGestureType = TYPE_SWIPE;
                            lastGesturesFrameIds.at(TYPE_SWIPE).second = currentFrame;
                            break;
                        }

                        // -------------------------- KEY_TAP Gesture ----------------------------
                        case Gesture::TYPE_KEY_TAP: {
                            if(((currentFrame-lastGesturesFrameIds.at(TYPE_KEY_TAP).second)/myListener.getCurrentFramesPerSecond())<1.0)
                            {
                                if(f_printLog.getValue())
                                    std::cout << "KEY TAP REJECTED !" << std::endl;
                                break;
                            }

                            KeyTapGesture tap = gesture;
                            gestureType.setValue(TYPE_KEY_TAP);
                            gesturePosition.setValue(Vec3d(tap.position().toFloatPointer()));
                            gestureDirection.setValue(Vec3d(tap.direction().toFloatPointer()));
                            scrollDirection.setValue(0);

                            if(f_printLog.getValue())
                                std::cout << "KEY_TAP gesture detected"<<std::endl;

                            lastGestureType = TYPE_KEY_TAP;
                            lastGesturesFrameIds.at(TYPE_KEY_TAP).second = currentFrame;
                            break;
                        }

                        // -------------------------- SCREEN_TAP Gesture ----------------------------
                        case Gesture::TYPE_SCREEN_TAP: {
                            if(((currentFrame-lastGesturesFrameIds.at(TYPE_SCREEN_TAP).second)/myListener.getCurrentFramesPerSecond())<1.0)
                            {
                                if(f_printLog.getValue())
                                    std::cout << "SCREEN TAP REJECTED !" << std::endl;
                                break;
                            }

                            ScreenTapGesture screentap = gesture;
                            gestureType.setValue(TYPE_SCREEN_TAP);
                            gesturePosition.setValue(Vec3d(screentap.position().toFloatPointer()));
                            gestureDirection.setValue(Vec3d(screentap.direction().toFloatPointer()));
                            scrollDirection.setValue(0);

                            if(f_printLog.getValue())
                                std::cout << "SCREEN_TAP gesture detected"<<std::endl;

                            lastGestureType = TYPE_SCREEN_TAP;
                            lastGesturesFrameIds.at(TYPE_SCREEN_TAP).second = currentFrame;
                            break;
                        }


                        // -------------------------- Unknown Gesture ----------------------------
                        default:
                            break;

                    } // switch

                    lastGestureFrameId = currentFrame;
                } // isValid & stateSTOP
            } // for loop
        }
    } // AnimatedBeginEvent
}

int LeapMotionDriverClass = core::RegisterObject("LeapMotion device driver")
.add< LeapMotionDriver >();

SOFA_DECL_CLASS(LeapMotionDriver)


} // namespace controller

} // namespace component

} // namespace sofa

