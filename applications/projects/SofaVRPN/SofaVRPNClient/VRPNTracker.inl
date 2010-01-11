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

#include <sofa/defaulttype/Quat.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa::defaulttype;

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
    	points.clear();
    	//points.resize(3);
    	Coord p0(0.0, 0.0, 1.0);
    	Coord p1(1.0, 0.0, 1.0);
    	Coord p2(0.0, 0.0, -1.0);
    	Quat qX(Vec3d(1.0,0.0,0.0), angleX);
    	Quat qY(Vec3d(0.0,1.0,0.0), angleY);
    	Quat qZ(Vec3d(0.0,0.0,1.0), angleZ);
    	p0 = qX.rotate(p0);p1 = qX.rotate(p1);p2 = qX.rotate(p2);
    	p0 = qY.rotate(p0);p1 = qY.rotate(p1);p2 = qY.rotate(p2);
    	p0 = qZ.rotate(p0);p1 = qZ.rotate(p1);p2 = qZ.rotate(p2);

    	points.push_back(p0);
    	points.push_back(p1);
    	points.push_back(p2);
    */

    /*
    for (unsigned int i=0 ; i<3 ;i++)
    {
    	for (unsigned int j=0 ; j<3 ;j++)
    	{
    		points[i][j] = (Real)rg.randomDouble(0, 10);
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
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        std::cout << angleX << std::endl;
        std::cout << angleY << std::endl;
        std::cout << angleZ << std::endl;
        std::cout << std::endl;

        switch(ev->getKey())
        {

        case 'A':
        case 'a':
            angleX -= M_PI/6;
            break;
        case 'Q':
        case 'q':
            angleX += M_PI/6;
            break;
        case 'Z':
        case 'z':
            angleY -= M_PI/6;
            break;
        case 'S':
        case 's':
            angleY += M_PI/6;
            break;
        case 'E':
        case 'e':
            angleZ -= M_PI/6;
            break;
        case 'D':
        case 'd':
            angleZ += M_PI/6;
            break;

        }
    }

}

}

}

#endif /* SOFAVRPNCLIENT_VRPNTRACKER_INL_ */
