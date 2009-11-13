/*
 * IRTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_IRTRACKER_INL_
#define SOFAVRPNCLIENT_IRTRACKER_INL_

#include <IRTracker.h>

#include <sofa/core/ObjectFactory.h>

namespace sofavrpn
{

namespace client
{

// 	  Wiimote
//    channel[0] = battery level (0-1)
//    channel[1] = gravity X vector calculation (1 = Earth gravity)
//    channel[2] = gravity Y vector calculation (1 = Earth gravity)
//    channel[3] = gravity Z vector calculation (1 = Earth gravity)
//    channel[4] = X of first sensor spot (0-1023, -1 if not seen)
//    channel[5] = Y of first sensor spot (0-767, -1 if not seen)
//    channel[6] = size of first sensor spot (0-15, -1 if not seen)
//    channel[7] = X of second sensor spot (0-1023, -1 if not seen)
//    channel[9] = Y of second sensor spot (0-767, -1 if not seen)
//    channel[9] = size of second sensor spot (0-15, -1 if not seen)
//    channel[10] = X of third sensor spot (0-1023, -1 if not seen)
//    channel[11] = Y of third sensor spot (0-767, -1 if not seen)
//    channel[12] = size of third sensor spot (0-15, -1 if not seen)
//    channel[13] = X of fourth sensor spot (0-1023, -1 if not seen)
//    channel[14] = Y of fourth sensor spot (0-767, -1 if not seen)
//    channel[15] = size of fourth sensor spot (0-15, -1 if not seen)
// and on the secondary controllers (skipping values to leave room for expansion)
//    channel[16] = nunchuck gravity X vector
//    channel[17] = nunchuck gravity Y vector
//    channel[18] = nunchuck gravity Z vector
//    channel[19] = nunchuck joystick angle
//    channel[20] = nunchuck joystick magnitude
//
//    channel[32] = classic L button
//    channel[33] = classic R button
//    channel[34] = classic L joystick angle
//    channel[35] = classic L joystick magnitude
//    channel[36] = classic R joystick angle
//    channel[37] = classic R joystick magnitude
//
//    channel[48] = guitar hero whammy bar
//    channel[49] = guitar hero joystick angle
//    channel[50] = guitar hero joystick magnitude

template<class Datatypes>
const double IRTracker<Datatypes>::WIIMOTE_X_ANGLE = 40;

template<class Datatypes>
const double IRTracker<Datatypes>::WIIMOTE_Y_ANGLE = 30;

template<class Datatypes>
const double IRTracker<Datatypes>::WIIMOTE_X_RESOLUTION = 1024;

template<class Datatypes>
const double IRTracker<Datatypes>::WIIMOTE_Y_RESOLUTION = 768;

template<class Datatypes>
IRTracker<Datatypes>::IRTracker()
    : f_leftDots(initData(&f_leftDots, "leftDots", "IR dots from L camera"))
    , f_rightDots(initData(&f_rightDots, "rightDots", "IR dots from R camera"))
    , f_distance(initData(&f_distance, "distance", "Distance between the 2 cameras"))
    , f_scale(initData(&f_scale, "scale", "Scale"))
    , f_points(initData(&f_points, "points", "Computed 3D Points"))
{
    addInput(&f_leftDots);
    addInput(&f_rightDots);
    addInput(&f_distance);

    addOutput(&f_points);;

    setDirtyValue();

}

template<class Datatypes>
IRTracker<Datatypes>::~IRTracker()
{
    // TODO Auto-generated destructor stub
}

//Frame is from user, not the wiimote
template<class Datatypes>
void IRTracker<Datatypes>::update()
{
    cleanDirty();

    sofa::helper::ReadAccessor< Data<VecCoord > > leftDots = f_leftDots;
    sofa::helper::ReadAccessor< Data<VecCoord > > rightDots = f_rightDots;
    const double &distanceCam = f_distance.getValue();
    const double &scale = f_scale.getValue();

    double XHalfAngleRad = M_PI*(WIIMOTE_X_ANGLE*0.5)/180.0;
    double YHalfAngleRad = M_PI*(WIIMOTE_Y_ANGLE*0.5)/180.0;

    sofa::helper::WriteAccessor< Data<VecCoord > > points = f_points;

    points.clear();
    points.resize(1);
    double minimalDistance = ((distanceCam*0.5)/tan(XHalfAngleRad));
    //std::cout << "Min Distance is " << minimalDistance << std::endl;
    //for (unsigned int i=0 ; i< 1 ;i++)
    for (unsigned int i=0 ; i< leftDots.size() && i < rightDots.size() ; i++)
    {
        Coord p;
        Coord ldot = leftDots[i];
        Coord rdot = rightDots[i];

        /*if (ldot[0] < 0.0 || ldot[1] < 0.0)
        	std::cout << "Dot " << i << " out of range of camera 1" << std::endl;
        if (rdot[0] < 0.0 || rdot[1] < 0.0)
        	std::cout << "Dot " << i << " out of range of camera 2" << std::endl;
        */

        //if we see the dot on 2 cameras
        //if (!((ldot[0] < 0.0 || ldot[1] < 0.0) && (rdot[0] < 0.0 || rdot[1] < 0.0)))
        if ((ldot[0] > 0.0 && ldot[1] > 0.0) && (rdot[0] > 0.0 && rdot[1] > 0.0))
        {
            //right or left ?
            if (ldot[0] < rdot[0])
            {
                Coord t = ldot;
                ldot = rdot;
                rdot = t;
            }

            double lxnormalized = (ldot[0]/(WIIMOTE_X_RESOLUTION/2)) - 1;
            double rxnormalized = (rdot[0]/(WIIMOTE_X_RESOLUTION/2)) - 1;
            double lynormalized = (ldot[1]/(WIIMOTE_Y_RESOLUTION/2)) - 1;

            p[2] =  distanceCam / ( (lxnormalized*tan(XHalfAngleRad) - rxnormalized*tan(XHalfAngleRad)));

            //X
            p[0] = (distanceCam*0.5) + rxnormalized*tan(XHalfAngleRad)*p[2];

            //Y
            p[1] = lynormalized*tan(YHalfAngleRad)*p[2];

            //std::cout << "Point " << i << " : " << p*100 << std::endl;

            points.push_back(p);
        }
        //else
        //	p = Coord(0.0,0.0,0.0);
    }
}

}

}

#endif //SOFAVRPNCLIENT_IRTRACKER_INL_

