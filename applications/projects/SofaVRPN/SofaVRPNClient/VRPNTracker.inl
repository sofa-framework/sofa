/*
 * VRPNTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#ifndef SOFAVRPNCLIENT_VRPNTRACKER_INL_
#define SOFAVRPNCLIENT_VRPNTRACKER_INL_

#include "VRPNTracker.h"

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofavrpn
{

namespace client
{

template<class DataTypes>
VRPNTracker<DataTypes>::VRPNTracker()
    : f_points(initData(&f_points, "points", "Points from Sensors"))
{
    // TODO Auto-generated constructor stub
    trackerData.data.resize(1);
    rg.initSeed( (long int) this );

}

template<class DataTypes>
VRPNTracker<DataTypes>::~VRPNTracker()
{
    // TODO Auto-generated destructor stub
}

template<class DataTypes>
bool VRPNTracker<DataTypes>::connectToServer()
{
    tkr = new vrpn_Tracker_Remote(deviceURL.c_str());
    tkr->register_change_handler((void*) &trackerData, handle_tracker);

    tkr->reset_origin();

    tkr->mainloop();

    return true;
}

template<class DataTypes>
void VRPNTracker<DataTypes>::update()
{
    sofa::helper::WriteAccessor< Data< VecCoord > > points = f_points;
    //std::cout << "read tracker " << this->getName() << std::endl;

    if (tkr)
    {
        //get infos
        trackerData.modified = false;
        tkr->mainloop();

        if(trackerData.modified)
        {
            points.clear();
            //if (points.size() < trackerData.data.size())
            points.resize(trackerData.data.size());

            for (unsigned int i=0 ; i<trackerData.data.size() ; i++)
            {
                for (unsigned int j=0 ; j<3 ; j++)
                {
                    points[i][j] = trackerData.data[i].pos[j];
                    //points[i][j] = (Real)rg.randomDouble(0, 10);
                    //TODO: quat
                }
            }
        }
    }
    /*
    	points.resize(5);
    	for (unsigned int i=0 ; i<5 ;i++)
    	{
    		for (unsigned int j=0 ; j<3 ;j++)
    		{
    			points[i][j] = i + j;
    			//TODO: quat
    		}
    	}
    */
    //std::cout << "POINTS :" << points << std::endl;
}

template<class DataTypes>
void VRPNTracker<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    update();
    /*if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
    	switch(ev->getKey())
    	{

    		case 'T':
    		case 't':
    			update();
    			break;
    	}
    }
    */
}

}

}

#endif /* SOFAVRPNCLIENT_VRPNTRACKER_INL_ */
