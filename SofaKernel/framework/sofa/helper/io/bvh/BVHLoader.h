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
#ifndef SOFA_HELPER_IO_BVH_BVHLOADER_H
#define SOFA_HELPER_IO_BVH_BVHLOADER_H

#include <sofa/helper/io/bvh/BVHJoint.h>
#include <sofa/helper/io/bvh/BVHMotion.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace io
{

namespace bvh
{

/**
*	This class defines a BVH File Loader
*	This files describe a hierarchical articulated model and also an associated motion
*	see http://www.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html for the file format specification
*/
class SOFA_HELPER_API BVHLoader
{
public:
    BVHLoader() {};
    virtual ~BVHLoader() {};

    BVHJoint *load(const char *filename);

private:
    BVHJoint *parseJoint(FILE *f, bool isEndSite=false, BVHJoint *parent=NULL);
    BVHOffset *parseOffset(FILE *f);
    BVHChannels *parseChannels(FILE *f);

    void parseMotion(FILE *f, BVHJoint *j);
    void setFrameTime(BVHJoint *j, double _frameTime);
    void parseFrames(BVHJoint *j, unsigned int frameIndex, FILE *f);
};

} // namespace bvh

} // namespace io

} // namespace helper

} // namespace sofa

#endif
