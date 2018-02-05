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
    , f_distance(initData(&f_distance, "distanceCamera", "Distance between the 2 cameras"))
    , f_distanceSide(initData(&f_distanceSide, "distanceSide", "Distance of a side"))
    , f_scale(initData(&f_scale, "scale", "Scale"))
    , f_points(initData(&f_points, "points", "Computed 3D Points"))
    , p_yErrorCoeff(initData(&p_yErrorCoeff, 1.0, "yErrorCoeff", "Y Error Coefficient"))
    , p_sideErrorCoeff(initData(&p_sideErrorCoeff, 1.0, "sideErrorCoeff", "Side Error Coefficient"))
    , p_realSideErrorCoeff(initData(&p_realSideErrorCoeff, 1.0, "realSideErrorCoeff", "Real Side Error Coefficient"))
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

template<class Datatypes>
typename Datatypes::Coord IRTracker<Datatypes>::get3DPoint(double lx, double ly, double rx, double /* ry */)
{
    //right or left ?
//	if (lx < rx)
//	{
//		Coord t = ldot;
//		ldot = rdot;
//		rdot = t;
//	}
    Coord p;
    const double &distanceCam = f_distance.getValue();

    double XHalfAngleRad = M_PI*(WIIMOTE_X_ANGLE*0.5)/180.0;
    double YHalfAngleRad = M_PI*(WIIMOTE_Y_ANGLE*0.5)/180.0;

    double lxnormalized = (lx/(WIIMOTE_X_RESOLUTION/2)) - 1;
    double rxnormalized = (rx/(WIIMOTE_X_RESOLUTION/2)) - 1;
    double lynormalized = (ly/(WIIMOTE_Y_RESOLUTION/2)) - 1;

    p[2] =  distanceCam / ( (lxnormalized*tan(XHalfAngleRad) - rxnormalized*tan(XHalfAngleRad)));

    //X
    p[0] = (distanceCam*0.5) + rxnormalized*tan(XHalfAngleRad)*p[2];

    //Y
    p[1] = lynormalized*tan(YHalfAngleRad)*p[2];

    return p;
}

//Frame is from user, not the wiimote
template<class Datatypes>
void IRTracker<Datatypes>::update()
{
    cleanDirty();

    sofa::helper::ReadAccessor< sofa::core::objectmodel::Data<VecCoord > > leftDots = f_leftDots;
    sofa::helper::ReadAccessor< sofa::core::objectmodel::Data<VecCoord > > rightDots = f_rightDots;

    //const double &scale = f_scale.getValue();
    const double &distanceSide = f_distanceSide.getValue();

    sofa::helper::WriteAccessor< sofa::core::objectmodel::Data<VecCoord > > points = f_points;

    if (leftDots.size() != rightDots.size())
    {
        std::cout << "Left != Right"<< std::endl;
        return ;
    }

    unsigned int numberOfVisibleDots = 0;
    for (unsigned int i=0 ; i<leftDots.size() ; i++)
    {
        if (leftDots[i][0] > 0.0 && rightDots[i][0] > 0.0)
            numberOfVisibleDots++;
    }

    if (numberOfVisibleDots != 3)
    {
        std::cout << "Detect only "<< numberOfVisibleDots << " dots !" << std::endl;
        return ;
    }

    points.clear();

    unsigned int table[36] =
    {
        0, 0, 1, 1, 2, 2,
        0, 0, 1, 2, 2, 1,
        0, 1, 1, 2, 2, 0,
        0, 1, 1, 0, 2, 2,
        0, 2, 1, 0, 2, 1,
        0, 2, 1, 1, 2, 0,
    };

    double minError = 999999.0;
    VecCoord bestPoints;
    //Guess the correct mapping between dots
    for (unsigned int i=0 ; i< 6 ; i++)
    {
        VecCoord tempPoints;
        bool valid = true;
        double yError = 0.0;
        double sideError = 0.0;
        double realSideError = 0.0;

        for (unsigned int j=0 ; j<numberOfVisibleDots && valid; j++)
        {
            Coord ldot = leftDots [table[i*6+j*2]];
            Coord rdot = rightDots[table[i*6+j*2+1]];

            //valid if the right dot's X < left dot's X
            if ( (valid = (ldot[0] > rdot[0])) )
            {
                //compute Y error
                yError += abs(ldot[1] - rdot[1]);
                //get 3D position
                tempPoints.push_back(get3DPoint(ldot[0], ldot[1], rdot[0], rdot[1]));
            }
        }

        if (valid)
        {
            std::vector<double> lengths;
            //get the (possible) top point
            double length01 = (tempPoints[1] - tempPoints[0]).norm();
            lengths.push_back(length01);
            double length02 = (tempPoints[2] - tempPoints[0]).norm();
            lengths.push_back(length02);
            double length12 = (tempPoints[2] - tempPoints[1]).norm();
            lengths.push_back(length12);

            std::sort(lengths.begin(), lengths.end());
            //first is the less so it is not a side
            sideError = lengths[2]-lengths[1];
            realSideError = abs(distanceSide - (lengths[2]/lengths[1])*0.5);

            if (minError > yError + sideError + realSideError)
            {
                minError = p_yErrorCoeff.getValue()*yError
                        + p_sideErrorCoeff.getValue()*sideError
                        + p_realSideErrorCoeff.getValue()*realSideError;
                bestPoints = tempPoints;
            }
        }

    }
    for (unsigned int i=0 ; i<bestPoints.size() ; i++)
        points.push_back(bestPoints[i]);

}


}

}

#endif //SOFAVRPNCLIENT_IRTRACKER_INL_

