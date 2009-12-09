/*
 * ToolFinder.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_TOOLFINDER_H_
#define SOFAVRPNCLIENT_TOOLFINDER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/DataEngine.h>

#include <VRPNDevice.h>

#include <vrpn/vrpn_Analog.h>

namespace sofavrpn
{

namespace client
{

template<class DataTypes>
class ToolFinder : public virtual sofa::core::objectmodel::BaseObject, public virtual sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ToolFinder, DataTypes), sofa::core::objectmodel::BaseObject);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename sofa::defaulttype::RigidTypes::Coord RPoint;
    typedef typename sofa::defaulttype::RigidTypes::Coord RCoord;

    //input
    Data<VecCoord > f_points;
    Data<Real> f_distance;

    //output
    Data<Coord> f_leftPoint;
    Data<Coord> f_rightPoint;
    Data<RCoord> f_topPoint;
    Data<Real> f_angle;
    Data<sofa::helper::vector<double> > f_angleArticulated;

    ToolFinder();
    virtual ~ToolFinder();

//	void init();
//	void reinit();
    void update();

private:

};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_TOOLFINDER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API ToolFinder<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API ToolFinder<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_TOOLFINDER_H_ */
