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

namespace sofavrpn
{

namespace client
{

struct VRPNTrackerData
{
    sofa::helper::vector<vrpn_TRACKERCB> data;
    bool modified;
};

void handle_tracker(void *userdata, const vrpn_TRACKERCB t);

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
    Data<VecCoord> f_points;

    Data<Real> p_dx, p_dy, p_dz;
    Data<Real> p_scale;
    Data<bool> p_nullPoint;


    VRPNTrackerData trackerData;
    vrpn_Tracker_Remote* tkr;
    sofa::helper::RandomGenerator rg;
    bool connectToServer();
    void update();

    void handleEvent(sofa::core::objectmodel::Event* event);
    //DEBUG
    double angleX, angleY, angleZ;
};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_VRPNTRACKER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_VRPNTRACKER_H_ */
