/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <sofa/helper/RandomGenerator.h>

#include <VRPNDevice.h>

#include <vrpn/vrpn_Tracker.h>
#include <memory>

namespace sofavrpn
{

namespace client
{

template<class DataTypes>
class VRPNTracker :  public virtual VRPNDevice
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VRPNTracker, DataTypes), VRPNDevice);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;

    typedef typename DataTypes::VecCoord VecCoord;

    VRPNTracker();
    virtual ~VRPNTracker();

//	void init();
//	void reinit();

private:
    sofa::core::objectmodel::Data<VecCoord> f_points;

    sofa::core::objectmodel::Data<Real> p_dx, p_dy, p_dz;
    sofa::core::objectmodel::Data<Real> p_scale;
    sofa::core::objectmodel::Data<bool> p_nullPoint;

    std::auto_ptr<vrpn_Tracker_Remote> tkr;
    sofa::helper::RandomGenerator rg;

    bool connectToServer();
    void update();
    void updateCallback(const vrpn_TRACKERCB& t);

    static void handle_tracker(void* userdata, const vrpn_TRACKERCB t);
    void handleEvent(sofa::core::objectmodel::Event* event);
    //DEBUG
    double angleX, angleY, angleZ;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFAVRPNCLIENT_VRPNTRACKER_CPP_)
#ifndef SOFA_FLOAT
extern template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_VRPNTRACKER_H_ */
