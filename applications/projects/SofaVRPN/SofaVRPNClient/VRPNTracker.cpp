/*
 * VRPNTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#include "VRPNTracker.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofavrpn
{

namespace client
{

int VRPNTrackerClass = sofa::core::RegisterObject("VRPN Tracker")
        .add< VRPNTracker >();

SOFA_DECL_CLASS(VRPNTracker)

void handle_tracker(void *userdata, const vrpn_TRACKERCB t)
{
    printf("Sensor %3d is now at (%11f, %11f, %11f, %11f, %11f, %11f, %11f)           \r",
            t.sensor, t.pos[0], t.pos[1], t.pos[2],
            t.quat[0], t.quat[1], t.quat[2], t.quat[3]);
    fflush(stdout);
}

VRPNTracker::VRPNTracker()
{
    // TODO Auto-generated constructor stub

}

VRPNTracker::~VRPNTracker()
{
    // TODO Auto-generated destructor stub
}

bool VRPNTracker::connectToServer()
{
    tkr = new vrpn_Tracker_Remote(deviceURL.c_str());
    tkr->register_change_handler(NULL, handle_tracker);

    tkr->reset_origin();

    //main interactive loop

    // Let the tracker do its thing
    tkr->mainloop();

    return true;
}

void VRPNTracker::update()
{
//	if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
//	{
//		switch(ev->getKey())
//		{
//
//			case 'T':
//			case 't':
//				std::cout << "Tracker : " << std::endl;
//				break;
//		}
//	}
}

}

}
