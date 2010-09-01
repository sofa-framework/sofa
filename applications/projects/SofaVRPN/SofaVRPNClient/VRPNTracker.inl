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
    , p_dx(initData(&p_dx, (Real) 0.0, "dx", "Translation along X axis"))
    , p_dy(initData(&p_dy, (Real) 0.0, "dy", "Translation along Y axis"))
    , p_dz(initData(&p_dz, (Real) 0.0, "dz", "Translation along Z axis"))
    , p_scale(initData(&p_scale, (Real) 1.0, "scale", "Scale (3 axis)"))
    , p_nullPoint(initData(&p_nullPoint, true, "nullPoint", "If not tracked, return a (0, 0, 0) point"))
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
            Coord pos;

            if (copyTrackerData.data.size() < 1)
            {
                pos[0] = 0.0;
                pos[1] = 0.0;
                pos[2] = 0.0;
                points.push_back(pos);
            }
            else
            {
                for (unsigned int i=0 ; i<copyTrackerData.data.size() ; i++)
                {

                    pos[0] = (copyTrackerData.data[i].pos[0])*p_scale.getValue() + p_dx.getValue();
                    pos[1] = (copyTrackerData.data[i].pos[1])*p_scale.getValue() + p_dy.getValue();
                    pos[2] = (copyTrackerData.data[i].pos[2])*p_scale.getValue() + p_dz.getValue();

                    Coord p(pos);
                    points[i] = p;
                }
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
