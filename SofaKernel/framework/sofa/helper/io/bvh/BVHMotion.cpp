/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/io/bvh/BVHMotion.h>
#include <sofa/helper/logging/Messaging.h>

#include <iostream>

namespace sofa
{

namespace helper
{

namespace io
{

namespace bvh
{

void BVHMotion::init(double _fTime, unsigned int _fCount, unsigned int _fSize)
{
    frameTime = _fTime;
    frameCount = _fCount;

    frames.resize(frameCount);

    for (int i=0; i<frameCount; i++)
        frames[i].resize(_fSize);
}

void BVHMotion::debug(void)
{
    for (unsigned int i=0; i<frames.size(); i++)
    {
        std::stringstream tmpmsg;
        for (unsigned int j=0; j<frames[i].size(); j++)
            tmpmsg << frames[i][j] << " ";
        msg_info("BVHMotion") ;
    }
}

} // namespace bvh

} // namespace io

} // namespace helper

} // namespace sofa
