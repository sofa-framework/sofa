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
        VRPNTrackerData copyTrackerData(trackerData);

        if(copyTrackerData.modified)
        {
            points.clear();
            //if (points.size() < trackerData.data.size())
            points.resize(copyTrackerData.data.size());

            for (unsigned int i=0 ; i<copyTrackerData.data.size() ; i++)
            {
                Coord pos;
                for (unsigned int j=0 ; j<3 ; j++)
                {
                    pos[j] = copyTrackerData.data[i].pos[j];
                    //points[i][j] = (Real)rg.randomDouble(0, 10);
                    //TODO: quat
                }
                Coord p(pos);
                points[i] = p;
            }
        }
    }
    /*
    	points.clear();
    	//points.resize(3);

    	Coord p0(0.113883,-0.049363,0.125364);
    	Coord p1(0.080272,0.054685, 0.055034);
    	Coord p2(0.026031, 0.044880, 0.043384);

    	Coord bary = (p0+p1+p2)/3.0;

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
}

template<class DataTypes>
void VRPNTracker<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    update();
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        /*std::cout << angleX << std::endl;
        std::cout << angleY << std::endl;
        std::cout << angleZ << std::endl;
        std::cout << std::endl;
        */
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
    }

}

}

}

#endif /* SOFAVRPNCLIENT_VRPNTRACKER_INL_ */
