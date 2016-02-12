#include <iostream>
#include "MyListener.h"


using namespace Leap;

double t1,t2;
double curSysTimeSeconds;
void MyListener::onInit(const Controller& )
{
  std::cout << "Leap Motion device initialized" << std::endl;

  //timer.start();
  t1=timer.getTime();
  //m_fLastUpdateTimeSeconds = timer.getElapsedTimeInSec();//::highResolutionTicksToSeconds(Time::getHighResolutionTicks());
  m_fLastUpdateTimeSeconds = timer.getTime();
  m_fLastRenderTimeSeconds = m_fLastUpdateTimeSeconds;
}

void MyListener::onConnect(const Controller& controller )
{
  std::cout << "Connected to Leap Motion device" << std::endl;
  controller.enableGesture(Gesture::TYPE_CIRCLE);
  controller.enableGesture(Gesture::TYPE_KEY_TAP);
  controller.enableGesture(Gesture::TYPE_SCREEN_TAP);
  controller.enableGesture(Gesture::TYPE_SWIPE);
}

void MyListener::onDisconnect(const Controller& )
{
  std::cout << "Leap Motion device disconnected" << std::endl;
}

void MyListener::onExit(const Controller& )
{
  std::cout << "Exited" << std::endl;
}

bool IsOdd (Leap::Finger /*finger*/)
{
	return true;
}

void MyListener::update( Leap::Frame /*frame*/ )
{
    //double curSysTimeSeconds = timer.getElapsedTimeInSec();
    t2=timer.getTime();
    curSysTimeSeconds= t2-t1;
    float deltaTimeSeconds = static_cast<float>(curSysTimeSeconds - m_fLastUpdateTimeSeconds);
      
    m_fLastUpdateTimeSeconds = curSysTimeSeconds;
    float fUpdateDT = m_avgUpdateDeltaTime.AddSample(deltaTimeSeconds);
    fUpdateFPS = (fUpdateDT > 0) ? 1.0f/fUpdateDT : 0.0f;
}

void MyListener::onFrame(const Controller& controller)
{
    frame = controller.frame();
    update( frame );
    currentFramesPerSecond = fUpdateFPS;//100.0;//frame.currentFramesPerSecond();

    if (!frame.hands().isEmpty())
    {
	//Store Hands
	hand = frame.hands()[0];
	interactionBox = frame.interactionBox();

	//Store palm hand orientation from Leap Motion
	Vec3d palmNormal  = Vec3d(hand.palmNormal().toFloatPointer());
	palmNormal.normalize();
	Vec3d palmToTipDirection = Vec3d(hand.direction().toFloatPointer());
	palmToTipDirection.normalize();
	Vec3d vecForFrameCreation = Vec3d(palmNormal).cross(palmToTipDirection);
	vecForFrameCreation.normalize();
	palmOrientation = Quat();
	palmOrientation.fromFrame(palmToTipDirection, vecForFrameCreation, palmNormal);
	palmOrientation.normalize();
    }

    gestures = frame.gestures();
    translation = hand.translation(controller.frame(10));

    if(frame.hands().count()>1)
    {
            secondHand = frame.hands()[1];
    }
    else
    {
            secondHand = Hand().invalid();
    }
}

void MyListener::onFocusGained(const Controller& ) {
  std::cout << "Leap Motion device focus Gained" << std::endl;
}

void MyListener::onFocusLost(const Controller& ) {
  std::cout << "Leap Motion device focus Lost" << std::endl;
}
