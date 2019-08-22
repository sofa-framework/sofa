#include <iostream>
#include <RazerHydraListener.h>


//using namespace Leap;

/*void RazerHydraListener::onInit(const Controller& ) {
  std::cout << "Leap Motion device initialized" << std::endl;
}

void RazerHydraListener::onConnect(const Controller& controller ) {
  std::cout << "Connected to Leap Motion device" << std::endl;
  controller.enableGesture(Gesture::TYPE_CIRCLE);
  controller.enableGesture(Gesture::TYPE_KEY_TAP);
  controller.enableGesture(Gesture::TYPE_SCREEN_TAP);
  controller.enableGesture(Gesture::TYPE_SWIPE);
}

void RazerHydraListener::onDisconnect(const Controller& ) {
  std::cout << "Leap Motion device disconnected" << std::endl;
}

void RazerHydraListener::onExit(const Controller& ) {
  std::cout << "Exited" << std::endl;
}

bool IsOdd (Leap::Finger finger) {
	return true;
}

void RazerHydraListener::update( Leap::Frame frame )
{

}

void RazerHydraListener::onFrame(const Controller& controller) {
    frame = controller.frame();
	currentFramesPerSecond = 100.0;//frame.currentFramesPerSecond();

  if (!frame.hands().empty()) {
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

	if(frame.hands().count()>1){
		secondHand = frame.hands()[1];
	} else {
		secondHand = Hand().invalid();
	}

}

void RazerHydraListener::onFocusGained(const Controller& ) {
  std::cout << "Leap Motion device focus Gained" << std::endl;
}

void RazerHydraListener::onFocusLost(const Controller& ) {
  std::cout << "Leap Motion device focus Lost" << std::endl;
}*/