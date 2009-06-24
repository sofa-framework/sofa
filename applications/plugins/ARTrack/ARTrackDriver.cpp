/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <ARTrackDriver.h>
#include <sofa/core/ObjectFactory.h>
#include <cstring>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <ARTrackEvent.h>

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
{
}

void ARTrackDriver::init()
{
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

    port = 5000; /* local port for communications with the tracker */
    rport = 5002; /* remote port of the machine controlling the tracker */
    strcpy(ip_address, "192.168.10.2"); /* IP address of the machine controlling the tracker */

    if(!( dataARTrack.handle = dtracklib_init(port, ip_address, rport, UDPBUFSIZE, UDPTIMEOUT)))
    {
        std::cout << "dtracklib init error" << std::endl;
        std::cout << "check that the tracker is opened and started (listening)" << std::endl;
    }

    if(!dtracklib_send(dataARTrack.handle, DTRACKLIB_CMD_CAMERAS_AND_CALC_ON, 0))
    {
        std::cout << "dtracklib send command error" << std::endl;
    }


    if(!dtracklib_send(dataARTrack.handle, DTRACKLIB_CMD_SEND_DATA, 0))
    {
        std::cout << "dtracklib send command error" << std::endl;
    }
}

#ifdef WIN32
void ARTrackDriver::computeTracking(void *sarg)
#else
void* ARTrackDriver::computeTracking(void *sarg)
#endif
{
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

        Vector3 wirstCurrentPos, wirstfilteredPos;
        Mat3x3d T(Vector3(0.0,0.0,-1.0), Vector3(-1.0,0.0,0.0), Vector3(0.0,1.0,0.0));
        Mat3x3d wirstRotMat;

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                wirstRotMat(j,i) = arg->glove[0].rot[(i*3)+j];

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

        wirstCurrentPos = arg->glove[0].loc;

        wirstfilteredPos = (arg->wristInitPos + wirstCurrentPos)/2;

#ifdef WIN32
        WaitForSingleObject(arg->mutex, INFINITE);
#else
        pthread_mutex_lock( &arg->mutex );
#endif

        arg->wirstTranslation = wirstfilteredPos - arg->wristInitPos;
        for (unsigned int i=0; i<arg->angle_finger.size(); ++i)
            arg->angle_finger[i] = arg->glove[0].finger[i].anglephalanx[1];

#ifdef WIN32
        ReleaseMutex(arg->mutex);
#else
        pthread_mutex_unlock( &arg->mutex );
#endif
    }
#ifdef WIN32
    return;
#endif
}


void ARTrackDriver::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        core::objectmodel::ARTrackEvent aRTrackEvent(dataARTrack.wirstTranslation, dataARTrack.wristRotation, dataARTrack.angle_finger);
        this->getContext()->propagateEvent(&aRTrackEvent);
    }
}

} // namespace controller

} // namespace component

} // namespace sofa
