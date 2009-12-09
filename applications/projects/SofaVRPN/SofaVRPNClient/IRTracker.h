/*
 * IRTracker.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_IRTRACKER_H_
#define SOFAVRPNCLIENT_IRTRACKER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/DataEngine.h>

#include <VRPNDevice.h>

#include <vrpn/vrpn_Analog.h>

namespace sofavrpn
{

namespace client
{

template<class DataTypes>
class IRTracker : public virtual sofa::core::objectmodel::BaseObject, public virtual sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(IRTracker, DataTypes), sofa::core::objectmodel::BaseObject);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    static const double WIIMOTE_X_RESOLUTION;
    static const double WIIMOTE_Y_RESOLUTION;
    static const double WIIMOTE_X_ANGLE;
    static const double WIIMOTE_Y_ANGLE;

    IRTracker();
    virtual ~IRTracker();


    //input
    Data<VecCoord > f_leftDots;
    Data<VecCoord > f_rightDots;
    Data<Real> f_distance;
    Data<Real> f_distanceSide;
    Data<Real> f_scale;

    //output
    Data<VecCoord> f_points;

    //Parameters
    Data<double> p_yErrorCoeff, p_sideErrorCoeff, p_realSideErrorCoeff;

    void update();

    Coord get3DPoint(double lx, double ly, double rx, double ry);
private:

};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_IRTRACKER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API IRTracker<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API IRTracker<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_IRTRACKER_H_ */
