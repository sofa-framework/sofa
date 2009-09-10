/*
 * VRPNTracker.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef VRPNTRACKER_H_
#define VRPNTRACKER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

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
    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

#endif /* VRPNTRACKER_H_ */
