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
 * VRPNImager.cpp
 *
 *  Created on: 14 May 2010
 *      Author: peterlik
 */

#define SOFAVRPNCLIENT_VRPNTRACKER_CPP_

#include "VRPNImager.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vrpnclient_config.h>



namespace sofavrpn
{

namespace client
{
void handle_discarded_frames(void *, const vrpn_IMAGERDISCARDEDFRAMESCB info)
{
    printf("Server discarded %d frames\n", (int)(info.count));
}


void  VRPN_CALLBACK handle_description_message(void *userData, const struct timeval)
{
    VRPNImagerData* imagerData = (VRPNImagerData*) userData;
    std::cout << "Description message" << std::endl;
    // This method is different from other VRPN callbacks because it simply
    // reports that values have been filled in on the Imager_Remote class.
    // It does not report what the new values are, only the time at which
    // they changed.

    // If we have already heard the dimensions, then check and make sure they
    // have not changed.  Also ensure that there is at least one channel.  If
    // not, then print an error and exit.
    if (imagerData->got_dimensions)
    {
        if ( (imagerData->Xdim != imagerData->remote_imager->nCols()) || (imagerData->Ydim != imagerData->remote_imager->nRows()) )
        {
            fprintf(stderr,"Error -- different image dimensions reported\n");
            exit(-1);
        }
        if (imagerData->remote_imager->nChannels() <= 0)
        {
            fprintf(stderr,"Error -- No channels to display!\n");
            exit(-1);
        }
    }

    // Record that the dimensions are filled in.  Fill in the globals needed
    // to store them.
    imagerData->Xdim = imagerData->remote_imager->nCols();
    imagerData->Ydim = imagerData->remote_imager->nRows();
    imagerData->got_dimensions = true;
}

void  VRPN_CALLBACK handle_region_change(void *userData, const vrpn_IMAGERREGIONCB info)
{
    VRPNImagerData* imagerData = (VRPNImagerData*) userData;

    const vrpn_Imager_Region  *region=info.region;
    const vrpn_Imager_Remote  *imager = imagerData->remote_imager;

    // Just leave things alone if we haven't set up the drawable things
    // yet.
    if (!imagerData->ready_for_region) { return; }

    // Copy pixels into the image buffer.
    // Flip the image over in Y so that the image coordinates
    // display correctly in OpenGL.
    // Figure out which color to put the data in depending on the name associated
    // with the channel index.  If it is one of "red", "green", or "blue" then put
    // it into that channel.
    if (strcmp(imager->channel(region->d_chanIndex)->name, "red") == 0)
    {
        region->decode_unscaled_region_using_base_pointer(imagerData->image+0, 3, 3*imagerData->Xdim, 0, imagerData->Ydim, true);
    }
    else if (strcmp(imager->channel(region->d_chanIndex)->name, "green") == 0)
    {
        region->decode_unscaled_region_using_base_pointer(imagerData->image+1, 3, 3*imagerData->Xdim, 0, imagerData->Ydim, true);
    }
    else if (strcmp(imager->channel(region->d_chanIndex)->name, "blue") == 0)
    {
        region->decode_unscaled_region_using_base_pointer(imagerData->image+2, 3, 3*imagerData->Xdim, 0, imagerData->Ydim, true);
    }
    else
    {
        region->decode_unscaled_region_using_base_pointer(imagerData->image, 3, 3*imagerData->Xdim, 0, imagerData->Ydim, true, 3);
    }

    unsigned char myOffset = sizeof(uint32_t);
    uint32_t numData;
    unsigned char line[imagerData->Xdim];

    for (unsigned i = 0; i < (unsigned)imagerData->Xdim; i++)
        line[i] = imagerData->image[3*i];

    memcpy(&numData, line, myOffset);
    //fprintf(stderr, "Amount of data = %d\n", numData);
    float *data  = (float *) calloc(numData, sizeof(float));
    memcpy(data, line+myOffset, numData*sizeof(float));

    for (unsigned i =  0; i < numData; i++)
    {
        imagerData->rigidPointData[i] = data[i];
        fprintf(stderr,"[%d] = %f ", i, imagerData->rigidPointData[i]);
    }
    fprintf(stderr,"\n");
    //std::cout << "rigidPoint = " << rigidPoint. << std::endl;
    free(data);
}

void  VRPN_CALLBACK handle_end_of_frame(void *userData,const struct _vrpn_IMAGERENDFRAMECB)
{
    VRPNImagerData* imagerData = (VRPNImagerData*) userData;
    // Tell Glut it is time to draw.  Make sure that we don't post the redisplay
    // operation more than once by checking to make sure that it has been handled
    // since the last time we posted it.  If we don't do this check, it gums
    // up the works with tons of redisplay requests and the program won't
    // even handle windows events.

    // NOTE: This exposes a race condition.  If more video messages arrive
    // before the frame-draw is executed, then we'll end up drawing some of
    // the new frame along with this one.  To make really sure there is not tearing,
    // double buffer: fill partial frames into one buffer and draw from the
    // most recent full frames in another buffer.  You could use an OpenGL texture
    // as the second buffer, sending each full frame into texture memory and
    // rendering a textured polygon.

    if (!imagerData->already_posted)
    {
        imagerData->already_posted = true;
        //glutPostRedisplay();
    }
}



using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(VRPNImager)

int VRPNImagerClass = sofa::core::RegisterObject("VRPN Imager")
#ifndef SOFA_FLOAT
        .add< VRPNImager<Vec3dTypes> >()
        .add< VRPNImager<Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< VRPNImager<Vec3fTypes> >()
        .add< VRPNImager<Rigid3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNImager<Vec3dTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNImager<Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNImager<Vec3fTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNImager<Rigid3fTypes>;
#endif //SOFA_DOUBLE

}

}
