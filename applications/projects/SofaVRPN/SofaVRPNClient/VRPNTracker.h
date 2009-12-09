/*
 * VRPNTracker.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_VRPNTRACKER_H_
#define SOFAVRPNCLIENT_VRPNTRACKER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/Event.h>

#include <VRPNDevice.h>

#include <vrpn/vrpn_Tracker.h>

namespace sofavrpn
{

namespace client
{

class VRPNTracker :  public virtual VRPNDevice
{
public:
    VRPNTracker();
    virtual ~VRPNTracker();

//	void init();
//	void reinit();

private:
    vrpn_Tracker_Remote* tkr;

    bool connectToServer();
    void update();

    void handleEvent(sofa::core::objectmodel::Event* event);
};

}

}

#endif /* SOFAVRPNCLIENT_VRPNTRACKER_H_ */
