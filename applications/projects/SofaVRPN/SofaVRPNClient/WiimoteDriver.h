/*
 * WiimoteDriver.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_WIIMOTEDRIVER_H_
#define SOFAVRPNCLIENT_WIIMOTEDRIVER_H_

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
class WiimoteDriver : public virtual sofa::core::objectmodel::BaseObject, public virtual sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(WiimoteDriver, DataTypes), sofa::core::objectmodel::BaseObject);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    static const unsigned int WIIMOTE_NUMBER_OF_IR_DOTS;

    //input
    Data<sofa::helper::vector<Real> > f_channels;

    //output
    Data<VecCoord> f_dots;

    //Parameters
    Data<bool> p_viewDots;

    WiimoteDriver();
    virtual ~WiimoteDriver();

//	void init();
//	void reinit();
    void update();
    void draw();

private:

};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_WIIMOTEDRIVER_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API WiimoteDriver<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API WiimoteDriver<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_WIIMOTEDRIVER_H_ */
