/*
 * ToolTracker.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_TOOLTRACKER_H_
#define SOFAVRPNCLIENT_TOOLTRACKER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Quat.h>
#include <VRPNDevice.h>

#include <vrpn/vrpn_Analog.h>

namespace sofavrpn
{

namespace client
{

/*
 * Find a tool given various parameters...
 *
 */

template<class DataTypes>
class ToolTracker : public virtual sofa::core::objectmodel::BaseObject, public virtual sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ToolTracker, DataTypes), sofa::core::objectmodel::BaseObject);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename sofa::defaulttype::RigidTypes::Coord RPoint;
    typedef typename sofa::defaulttype::RigidTypes::Coord RCoord;

    //input
    Data<VecCoord > f_points;
    //distances between each point for the given tool
    Data<sofa::helper::vector<double> > f_distances;

    //output
    Data<Coord> f_center;
    Data<sofa::defaulttype::Quat> f_orientation;
    //the same...
    Data<RCoord> f_rigidCenter;

    ToolTracker();
    virtual ~ToolTracker();

//	void init();
//	void reinit();
    void update();

private:

};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_TOOLTRACKER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API ToolTracker<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API ToolTracker<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_TOOLTRACKER_H_ */
