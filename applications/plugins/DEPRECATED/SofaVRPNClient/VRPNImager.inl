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
 * VRPNImager.cpp
 *
 *  Created on: 14 May 2010
 *      Author: peterlik, olsak
 */
#ifndef SOFAVRPNCLIENT_VRPNIMAGER_INL_
#define SOFAVRPNCLIENT_VRPNIMAGER_INL_

#include "VRPNImager.h"

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <sofa/defaulttype/Quat.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa::defaulttype;

template<class DataTypes>
VRPNImager<DataTypes>::VRPNImager()
//: f_positions( initData (&f_positions, "positions", "Positions (Vector of 3)") )
//, f_orientations( initData (&f_orientations, "orientations", "Orientations (Quaternion)") )
    : f_rigidPoint( initData (&f_rigidPoint, "rigid point", "RigidPoint")),
      f_scale( initData (&f_scale, "scale", "Scale"))

{
    addAlias(&f_rigidPoint,"rigidPosition");
    f_scale.setValue(1.0); // = 1.0;
    // TODO Auto-generated constructor stub
    /*trackerData.data.resize(1);
    rg.initSeed( (long int) this );*/

}

template<class DataTypes>
VRPNImager<DataTypes>::~VRPNImager()
{
    // TODO Auto-generated destructor stub
}


template<class DataTypes>
bool VRPNImager<DataTypes>::connectToServer()
{
    std::cout << "Connecting to imager server..." << std::endl;
    g_imager = new vrpn_Imager_Remote(deviceURL.c_str());
    imagerData.remote_imager = g_imager;
    g_imager->register_description_handler((void*) &imagerData, handle_description_message);
    g_imager->register_discarded_frames_handler((void*) &imagerData , handle_discarded_frames);
    g_imager->register_end_frame_handler((void*) &imagerData, handle_end_of_frame);
    g_imager->register_region_handler((void*) &imagerData, handle_region_change);

    std::cout << "Waiting to hear the image dimensions..." << std::endl;
    while (!imagerData.got_dimensions)
    {
        g_imager->mainloop();
        vrpn_SleepMsecs(1);
    }
    std::cout << "Connection established, dimensions " << imagerData.Xdim << " " << imagerData.Ydim << std::endl;
    g_imager->connectionPtr()->Jane_stop_this_crazy_thing(50);

    // Allocate memory for the image and clear it, so that things that don't get filled in will be black.
    if ( (imagerData.image = new unsigned char[imagerData.Xdim * imagerData.Ydim * 3]) == NULL)
    {
        std::cout << "Out of memory when allocating image!" << std::endl;
        return -1;
    }
    for (unsigned int i = 0; i < (unsigned)(imagerData.Xdim * imagerData.Ydim * 3); i++)
    {
        imagerData.image[i] = 0;
    }
    imagerData.ready_for_region = true;

    glGenTextures(1, &imageTextureID);

    return true;
}

template<class DataTypes>
void VRPNImager<DataTypes>::update()
{
    g_imager->mainloop();

    Point x;

    for (unsigned i=0; i < x.size(); i++)
        x[i] = imagerData.rigidPointData[i];
    //float temp = x[1];
    x[0]*= f_scale.getValue();
    x[1]= -x[1]*f_scale.getValue();
    x[2]= -x[2]*f_scale.getValue();


    if (x.size() > 3)
    {
        // process quaternion

        //x[0] *= 1;
        //x[1] *= -1;
        //x[2] *= -1;
        //x[2]; x[2]=temp;
        /*temp=x[5];
        x[5]=x[6];
        x[6]=temp;*/
        /*x[3] = 0;
        x[4] = 0;
        x[5] = 0;
        x[6] = 1;*/
        //x[3] = imagerData.rigidPointData[

        std::cout << "PointData = " <<  f_rigidPoint.getValue() << std::endl;
        Quat qt(x[3], x[4], x[5], x[6]);
        Vec3 currentAxisRotation;
        Real currentAngleRotation;
        qt.quatToAxis(currentAxisRotation, currentAngleRotation);

        //Quat qt2(currentAxisRotation*(-1), currentAngleRotation);
        currentAxisRotation[1] *= -1;
        currentAxisRotation[2] *= -1;

        /*Real ttemp;
        ttemp = currentAxisRotation[0];
        currentAxisRotation[0]=currentAxisRotation[2];
        currentAxisRotation[2]=ttemp;*/

        Quat qt2(currentAxisRotation, currentAngleRotation);

        x[3] = qt2[0];
        x[4] = qt2[1];
        x[5] = qt2[2];
        x[6] = qt2[3];
    }
    //std::cout << "Quat = " << qt << std::endl;
    //std::cout << "Euler = " << qt.toEulerVector() << std::endl;
    f_rigidPoint.setValue(x);


    /*sofa::helper::WriteAccessor< Data< VecCoord > > points = f_points;
    std::cout << "read tracker " << this->getName() << std::endl;

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

                for (unsigned int i=0 ; i<copyTrackerData.data.size() ;i++)
                {
                        Coord pos;
                        pos[0] = (copyTrackerData.data[i].pos[0])*p_scale.getValue() + p_dx.getValue();
                        pos[1] = (copyTrackerData.data[i].pos[1])*p_scale.getValue() + p_dy.getValue();
                        pos[2] = (copyTrackerData.data[i].pos[2])*p_scale.getValue() + p_dz.getValue();

                        Coord p(pos);
                        points[i] = p;
                }
        }
    }*/
}

template<class DataTypes>
void VRPNImager<DataTypes>::draw()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, imageTextureID);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_imager->nCols(), g_imager->nRows(), 0, GL_BGR, GL_UNSIGNED_BYTE, imagerData.image);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex4f(-1, -1, 0, 1);
    glTexCoord2i(1, 0);
    glVertex4f(1, -1, 0, 1);
    glTexCoord2i(1, 1);
    glVertex4f(1, 1, 0, 1);
    glTexCoord2i(0, 1);
    glVertex4f(-1, 1, 0, 1);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

template<class DataTypes>
void VRPNImager<DataTypes>::handleEvent(sofa::core::objectmodel::Event* /*event*/)
{

    update();

    /*if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {

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
    }*/

}

}

}

#endif /* SOFAVRPNCLIENT_VRPNIMAGER_INL_ */
