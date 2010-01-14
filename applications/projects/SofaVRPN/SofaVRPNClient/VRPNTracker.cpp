/*
 * VRPNTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#define SOFAVRPNCLIENT_VRPNTRACKER_CPP_

#include "VRPNTracker.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vrpnclient_config.h>

namespace sofavrpn
{

namespace client
{



void handle_tracker(void *userdata, const vrpn_TRACKERCB t)
{
    VRPNTrackerData* trackerData = (VRPNTrackerData*) userdata;

    trackerData->modified = true;

    if ((int)trackerData->data.size() < t.sensor + 1)
        trackerData->data.resize(t.sensor+1);

    trackerData->data[t.sensor].sensor = t.sensor;

    for (unsigned int i=0 ; i<3 ; i++)
    {
        trackerData->data[t.sensor].pos[i] = t.pos[i];
        trackerData->data[t.sensor].quat[i] = t.quat[i];
    }

    trackerData->data[t.sensor].quat[3] = t.quat[3];
}

template<>
void VRPNTracker<RigidTypes>::update()
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
                RigidTypes::Coord::Pos pos;
                RigidTypes::Coord::Rot rot;
                for (unsigned int j=0 ; j<3 ; j++)
                {
                    pos[j] = copyTrackerData.data[i].pos[j];
                    rot[j] = copyTrackerData.data[i].quat[j];
                }
                rot[3] = copyTrackerData.data[i].quat[3];
                Coord p;
                p.getCenter() = pos;
                p.getOrientation() = rot;
                points[i] = p;
            }
        }
    }
}

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(VRPNTracker)

int VRPNTrackerClass = sofa::core::RegisterObject("VRPN Tracker")
#ifndef SOFA_FLOAT
        .add< VRPNTracker<Vec3dTypes> >()
        .add< VRPNTracker<Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< VRPNTracker<Vec3fTypes> >()
        .add< VRPNTracker<RigidfTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3dTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3fTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Rigid3fTypes>;
#endif //SOFA_DOUBLE

}

}
