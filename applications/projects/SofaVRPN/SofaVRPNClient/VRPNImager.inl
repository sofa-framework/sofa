/*
 * VRPNImager.cpp
 *
 *  Created on: 14 May 2010
 *      Author: peterlik
 */
#ifndef SOFAVRPNCLIENT_VRPNIMAGER_INL_
#define SOFAVRPNCLIENT_VRPNIMAGER_INL_

#include "VRPNImager.h"

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <sofa/defaulttype/Quat.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa::defaulttype;

template<class DataTypes>
VRPNImager<DataTypes>::VRPNImager()
/*: f_points(initData(&f_points, "points", "Points from Sensors"))
, p_dx(initData(&p_dx, (Real) 0.0, "dx", "Translation along X axis"))
, p_dy(initData(&p_dy, (Real) 0.0, "dy", "Translation along Y axis"))
, p_dz(initData(&p_dz, (Real) 0.0, "dz", "Translation along Z axis"))
, p_scale(initData(&p_scale, (Real) 1.0, "scale", "Scale (3 axis)"))*/
{
    // TODO Auto-generated constructor stub
    /*trackerData.data.resize(1);
    rg.initSeed( (long int) this );*/

}

template<class DataTypes>
VRPNImager<DataTypes>::~VRPNImager()
{
    // TODO Auto-generated destructor stub
}


template<class DataTypes>
bool VRPNImager<DataTypes>::connectToServer()
{
    /*tkr = new vrpn_Tracker_Remote(deviceURL.c_str());
    tkr->register_change_handler((void*) &trackerData, handle_tracker);
    tkr->reset_origin();
    tkr->mainloop();*/

    std::cout << "Connecting to imager server..." << std::endl;
    g_imager = new vrpn_Imager_Remote(deviceURL.c_str());
    imagerData.remote_imager = g_imager;
    g_imager->register_description_handler((void*) &imagerData, handle_description_message);
    g_imager->register_discarded_frames_handler((void*) &imagerData , handle_discarded_frames);
    g_imager->register_end_frame_handler((void*) &imagerData, handle_end_of_frame);
    g_imager->register_region_handler((void*) &imagerData, handle_region_change);

    std::cout << "Waiting to hear the image dimensions..." << std::endl;
    while (!imagerData.got_dimensions)
    {
        g_imager->mainloop();
        vrpn_SleepMsecs(1);
    }
    std::cout << "Connection established, dimensions " << imagerData.Xdim << " " << imagerData.Ydim << std::endl;

    return true;
}

template<class DataTypes>
void VRPNImager<DataTypes>::update()
{
    /*sofa::helper::WriteAccessor< Data< VecCoord > > points = f_points;
    std::cout << "read tracker " << this->getName() << std::endl;

    if (tkr)
    {
    	//get infos
    	trackerData.modified = false;
    	tkr->mainloop();
    	VRPNTrackerData copyTrackerData(trackerData);

    	if(copyTrackerData.modified)
    	{
    		points.clear();
    		//if (points.size() < trackerData.data.size())
    		points.resize(copyTrackerData.data.size());

    		for (unsigned int i=0 ; i<copyTrackerData.data.size() ;i++)
    		{
    			Coord pos;
    			pos[0] = (copyTrackerData.data[i].pos[0])*p_scale.getValue() + p_dx.getValue();
    			pos[1] = (copyTrackerData.data[i].pos[1])*p_scale.getValue() + p_dy.getValue();
    			pos[2] = (copyTrackerData.data[i].pos[2])*p_scale.getValue() + p_dz.getValue();

    			Coord p(pos);
    			points[i] = p;
    		}
    	}
           }*/
}

template<class DataTypes>
void VRPNImager<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    std::cout << "handle event" << std::endl;
    /*update();
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {

    	double nb = 10.0;
    	switch(ev->getKey())
    	{

    		case 'A':
    		case 'a':
    			angleX -= M_PI/nb;
    			break;
    		case 'Q':
    		case 'q':
    			angleX += M_PI/nb;
    			break;
    		case 'Z':
    		case 'z':
    			angleY -= M_PI/nb;
    			break;
    		case 'S':
    		case 's':
    			angleY += M_PI/nb;
    			break;
    		case 'E':
    		case 'e':
    			angleZ -= M_PI/nb;
    			break;
    		case 'D':
    		case 'd':
    			angleZ += M_PI/nb;
    			break;

    	}
           }*/

}

}

}

#endif /* SOFAVRPNCLIENT_VRPNIMAGER_INL_ */
