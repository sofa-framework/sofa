#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <list>
#include <iostream>
#include <Leap.h>
#include <sofa/helper/system/thread/CTime.h>

using namespace Leap;
using namespace sofa::defaulttype;
using namespace sofa::helper::system::thread;


//LeapUtil RollingAverage Class to smooth Leap FPS
class RollingAverage
{
public:
  enum
  {
    kHistoryLength = 256
  };

public:
  RollingAverage() { Reset(); }

  void Reset() 
  {
    m_uiCurIndex  = 0;
    m_fNumSamples = 0.0f;
    m_fSum        = 0.0f;
    m_fAverage    = 0.0f;
    for ( int i = 0; i < kHistoryLength; m_afSamples[i++] = 0.0f );   
  }

  float AddSample( float fSample ) 
  {
    m_fNumSamples =   std::min( (m_fNumSamples + 1.0f), static_cast<float>(kHistoryLength) );
    m_fSum        -=  m_afSamples[m_uiCurIndex];
    m_fSum        +=  fSample;
    m_fAverage    =   m_fSum * (1.0f/m_fNumSamples);

    m_afSamples[m_uiCurIndex] = fSample;

    m_uiCurIndex = static_cast<uint32_t>((m_uiCurIndex + 1) % kHistoryLength);

    return m_fAverage;
  }

  float     GetAverage()    const { return m_fAverage; }
  uint32_t  GetNumSamples() const { return static_cast<uint32_t>(m_fNumSamples); }

  /// index 0 is the oldest sample, index kHistorySize - 1 is the newest.
  float     operator[](uint32_t uiIdx) const { return m_afSamples[(m_uiCurIndex + uiIdx) % kHistoryLength]; }

private:
  uint32_t      m_uiCurIndex;
  float         m_fNumSamples;
  float         m_fSum;
  float         m_fAverage;
  float         m_afSamples[kHistoryLength];
};

class MyListener : public Listener {
  public:

    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
    virtual void onFocusGained(const Controller&);
    virtual void onFocusLost(const Controller&);
    void update( Leap::Frame frame );

    int64_t     getFrameID() { return frame.id(); }
    Vector      getPalmPosition() { return hand.palmPosition(); }
    Vector      getStabilizedPalmPosition() { return hand.palmPosition(); }// hand.stabilizedPalmPosition(); }
    Vector      getPalmNormal() { return hand.palmNormal(); }
    Quat        getPalmRotation() { return palmOrientation; }
    FingerList  getFingers() { return hand.fingers(); }
    FingerList  getSecondHandFingers() { return secondHand.fingers(); }
    Hand        getHand() { return hand; }
    Hand        getSecondHand() { return secondHand; }
    Vector      getHandSphereCenter() { return hand.sphereCenter(); }
    float       getHandSphereRadius() { return hand.sphereRadius(); }
    GestureList getGestures() { return gestures; }
    Vector      getTranslation() { return translation; }
    InteractionBox getInteractionBox() { return interactionBox; }
    float	getFPS() { return fUpdateFPS;}
    float	getCurrentFramesPerSecond() {	return currentFramesPerSecond; }

protected:
    Frame frame;
    float currentFramesPerSecond;
    InteractionBox interactionBox;
    Hand hand, secondHand;
    Quat palmOrientation;
    GestureList gestures;
    Vector translation;

private:
    sofa::helper::system::thread::CTime timer;
    float  fUpdateFPS;
    RollingAverage m_avgUpdateDeltaTime;
    double	m_fLastUpdateTimeSeconds;
    double m_fLastRenderTimeSeconds;
};
