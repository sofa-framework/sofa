/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "ARTrackDriver.h"
#include <sofa/core/ObjectFactory.h>
#include <cstring>
#include <sofa/simulation/AnimateBeginEvent.h>
#include "ARTrackEvent.h"

namespace sofa
{

namespace component
{

namespace controller
{

SOFA_DECL_CLASS(ARTrackDriver)

int ARTrackDriverClass = core::RegisterObject("Driver for ARTrack system")
        .add< ARTrackDriver >()
        ;


ARTrackDriver::ARTrackDriver()
    : aRTrackScale( initData(&aRTrackScale,double(1.0),"aRTrackScale","ARTrack scale") )
    , localTrackerPos( initData(&localTrackerPos,Vector3(0,0,0),"localTrackerPos","Local tracker position") )
    , scaleAngleFinger( initData(&scaleAngleFinger,double(0.2),"scaleAngleFinger","Angle Finger scale") )
{
}

void ARTrackDriver::reinit()
{
    dataARTrack.aRTrackScale = aRTrackScale.getValue();
    dataARTrack.localTrackerPos = localTrackerPos.getValue();
}

void ARTrackDriver::init()
{
    dataARTrack.aRTrackScale = aRTrackScale.getValue();
    dataARTrack.localTrackerPos = localTrackerPos.getValue();

    initARTrack();

    if(!dtracklib_receive(dataARTrack.handle,&dataARTrack.framenr, &dataARTrack.timestamp,NULL, NULL, 0, 0,NULL, 0, 0,NULL, 0, 0,
            &dataARTrack.nmarker, dataARTrack.marker, MAX_NMARKER,&dataARTrack.nglove, dataARTrack.glove, MAX_NGLOVE))
    {
        if(dtracklib_timeout(dataARTrack.handle))
            std::cout << "--- timeout while waiting for udp data" << std::endl;
        if(dtracklib_udperror(dataARTrack.handle))
            std::cout << "--- error while receiving udp data" << std::endl;
        if(dtracklib_parseerror(dataARTrack.handle))
            std::cout << "--- error while parsing udp data" << std::endl;
    }

    dataARTrack.wristInitPos = dataARTrack.glove[0].loc;

    for (unsigned int i=0; i<dataARTrack.rest_angle_finger.size(); ++i)
        dataARTrack.rest_angle_finger[i] = dataARTrack.glove[0].finger[i].anglephalanx[1];
//    for (unsigned int i=0; i<dataARTrack.fingersInitPos.size(); ++i)
//		dataARTrack.fingersInitPos[i] = dataARTrack.glove[0].finger[i].loc;


#ifdef WIN32
    threadID = _beginthread( ARTrackDriver::computeTracking, 0, &dataARTrack);
    dataARTrack.mutex = CreateMutex( NULL, FALSE, NULL);
#else
    pthread_create( &threadID, NULL, ARTrackDriver::computeTracking, &dataARTrack);
    pthread_mutex_init(&dataARTrack.mutex, NULL);
#endif


}

void ARTrackDriver::initARTrack()
{
    int port, rport;
    char ip_address[16];

    std::cout<<" INIT TRACK "<<std::endl;

    port = 5000; /* local port for communications with the tracker */
    rport = 5002; /* remote port of the machine controlling the tracker */
    strcpy(ip_address, "192.168.1.3"); /* IP address of the machine controlling the tracker */

    if(!( dataARTrack.handle = dtracklib_init(port, ip_address, rport, UDPBUFSIZE, UDPTIMEOUT)))
    {
        std::cout << "dtracklib init error" << std::endl;
        std::cout << "check that the tracker is opened and started (listening)" << std::endl;
    }
    /*
        if(!dtracklib_send(dataARTrack.handle, DTRACKLIB_CMD_CAMERAS_AND_CALC_ON, 0))
        {
            std::cout << "dtracklib send command error" << std::endl;
        }


        if(!dtracklib_send(dataARTrack.handle, DTRACKLIB_CMD_SEND_DATA, 0))
        {
            std::cout << "dtracklib send command error" << std::endl;
        }
    	*/
}

#ifdef WIN32
void ARTrackDriver::computeTracking(void *sarg)
#else
void* ARTrackDriver::computeTracking(void *sarg)
#endif
{
    std::cout<<"computeTracking"<<std::endl;
    Vector3 wirstCurrentPos, wirstfilteredPos, tempPos;
    Mat3x3d T(Vector3(0.0,0.0,-1.0), Vector3(-1.0,0.0,0.0), Vector3(0.0,1.0,0.0));
    Mat3x3d wirstRotMat;
    sofa::helper::fixed_array<Vector3,3> fingersLocalPos;

    dataARTrackClass *arg = (dataARTrackClass*)sarg;

    while(true)
    {
        if(!dtracklib_receive(arg->handle,&arg->framenr, &arg->timestamp,NULL, NULL, 0, 0,NULL, 0, 0,NULL, 0, 0,
                &arg->nmarker, arg->marker, MAX_NMARKER,&arg->nglove, arg->glove, MAX_NGLOVE))
        {
            if(dtracklib_timeout(arg->handle))
                std::cout << "--- timeout while waiting for udp data" << std::endl;
            if(dtracklib_udperror(arg->handle))
                std::cout << "--- error while receiving udp data" << std::endl;
            if(dtracklib_parseerror(arg->handle))
                std::cout << "--- error while parsing udp data" << std::endl;
        }
        int glove_id;
        if(arg->glove[0].lr)
            glove_id=0;
        else
            glove_id=1;



        for (int i=0; i<3; i++)
        {
            for (int j=0; j<3; j++)
                wirstRotMat(j,i) = arg->glove[glove_id].rot[(i*3)+j];
        }


        wirstRotMat = wirstRotMat * T;

#ifdef WIN32

        WaitForSingleObject(arg->mutex, INFINITE);

#else
        pthread_mutex_lock( &arg->mutex );
#endif
        arg->wristRotation.fromMatrix(wirstRotMat);

#ifdef WIN32
        ReleaseMutex(arg->mutex);

#else
        pthread_mutex_unlock( &arg->mutex );
#endif

        wirstCurrentPos = arg->glove[glove_id].loc;



        wirstfilteredPos = wirstCurrentPos ;

        //std::cout<<"wirstfilteredPos :"<<wirstfilteredPos<<std::endl;

#ifdef WIN32

        WaitForSingleObject(arg->mutex, INFINITE);
#else
        pthread_mutex_lock( &arg->mutex );
#endif

        arg->wirstTranslation = wirstfilteredPos - arg->wristInitPos;



        for (unsigned int i=0; i<arg->angle_finger.size(); ++i)
        {
            std::cout<<"angle finger["<<i<<"] = "<<arg->glove[0].finger[i].anglephalanx[1]<<std::endl;
            arg->angle_finger[i] = (arg->glove[glove_id].finger[i].anglephalanx[1] - arg->rest_angle_finger[i]) * 0.017444; // *pi/180
            arg->angle_finger[i] *= 0.2;
            arg->angle_finger[i] -= 1.5;
        }


        for (unsigned int i=0; i<fingersLocalPos.size(); ++i)
        {
            fingersLocalPos[i] = arg->glove[glove_id].finger[i].loc;
            tempPos[0] = -fingersLocalPos[i][1];
            tempPos[1] = fingersLocalPos[i][2];
            tempPos[2] = -fingersLocalPos[i][0];
            tempPos *= arg->aRTrackScale;
            tempPos += arg->localTrackerPos;
            arg->fingersGlobalPos[i] = arg->wristRotation.rotate(tempPos) + arg->wirstTranslation;
        }

#ifdef WIN32
        ReleaseMutex(arg->mutex);

#else
        pthread_mutex_unlock( &arg->mutex );
#endif
    }
#ifdef WIN32
    return;
#else
    return NULL;
#endif
}



void ARTrackDriver::handleEvent(core::objectmodel::Event *event)
{
    //std::cout<<"ARTrack handleEvent "<<std::endl;
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        std::cout<<"ARTrack AnimateBeginEvent "<<std::endl;
        core::objectmodel::ARTrackEvent aRTrackEvent(dataARTrack.wirstTranslation, dataARTrack.wristRotation, dataARTrack.angle_finger, dataARTrack.fingersGlobalPos);

        this->getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &aRTrackEvent);
    }
}

} // namespace controller

} // namespace component

} // namespace sofa
