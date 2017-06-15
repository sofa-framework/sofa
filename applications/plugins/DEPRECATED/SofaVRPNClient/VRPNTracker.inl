/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Quat.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa::defaulttype;

template<typename R>
void setCoordFromTRACKERCB(Vec<3, R>& coord, const vrpn_TRACKERCB& t, R scale, R dx, R dy, R dz)
{
    coord.x() = t.pos[0] * scale + dx;
    coord.y() = t.pos[1] * scale + dy;
    coord.z() = t.pos[2] * scale + dz;
}

template<typename R>
void setCoordFromTRACKERCB(RigidCoord<3, R>& coord, const vrpn_TRACKERCB& t, R scale, R dx, R dy, R dz)
{
    setCoordFromTRACKERCB(coord.getCenter(), t, scale, dx, dy, dz);

    typename RigidCoord<3, R>::Rot& rot = coord.getOrientation();
    for(unsigned i = 0; i < 4; ++i)
    {
        rot[i] = t.quat[i];
    }
}

template<class DataTypes>
VRPNTracker<DataTypes>::VRPNTracker()
    : f_points(initData(&f_points, "points", "Points from Sensors"))
    , p_dx(initData(&p_dx, (Real) 0.0, "dx", "Translation along X axis"))
    , p_dy(initData(&p_dy, (Real) 0.0, "dy", "Translation along Y axis"))
    , p_dz(initData(&p_dz, (Real) 0.0, "dz", "Translation along Z axis"))
    , p_scale(initData(&p_scale, (Real) 1.0, "scale", "Scale (3 axis)"))
    , p_nullPoint(initData(&p_nullPoint, false, "nullPoint", "If not tracked, return a (0, 0, 0) point"))
{
    // TODO Auto-generated constructor stub
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
    tkr.reset(new vrpn_Tracker_Remote(deviceURL.c_str()));
    tkr->register_change_handler(static_cast<void*>(this), handle_tracker);

    tkr->reset_origin();

    tkr->mainloop();

    return true;
}

template<class DataTypes>
void VRPNTracker<DataTypes>::update()
{
    if (tkr.get() != 0)
    {
        tkr->mainloop();
    }
}

template<class DataTypes>
void VRPNTracker<DataTypes>::updateCallback(const vrpn_TRACKERCB& t)
{
    sofa::helper::WriteAccessor< sofa::core::objectmodel::Data< VecCoord > > points = f_points;

    if(points.size() < static_cast<unsigned>(t.sensor + 1))
    {
        points.resize(t.sensor + 1);
    }

    if(t.sensor >= 0)
    {
        setCoordFromTRACKERCB(points[t.sensor], t, p_scale.getValue(), p_dx.getValue(), p_dy.getValue(), p_dz.getValue());
    }
}

template<class DataTypes>
void VRPNTracker<DataTypes>::handle_tracker(void* userdata, const vrpn_TRACKERCB t)
{
    VRPNTracker<DataTypes>* tracker = reinterpret_cast<VRPNTracker<DataTypes>*>(userdata);
    tracker->updateCallback(t);
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
