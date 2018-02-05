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

#include "StereoCamera.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <iostream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(StereoCamera)

int StereoCameraClass = core::RegisterObject("StereoCamera")
        .add< StereoCamera >()
        ;


StereoCamera::StereoCamera()
    : _stereoEnabled(initData(&_stereoEnabled, true, "enabled", "Is the stereo mode initially enabled?"))
    , _stereoMode(initData(&_stereoMode, "mode", "Stereo Mode: STEREO_AUTO = 0, STEREO_INTERLACED = 1, STEREO_FRAME_PACKING = 2, STEREO_SIDE_BY_SIDE = 3, STEREO_TOP_BOTTOM = 4, STEREO_SIDE_BY_SIDE_HALF = 5, STEREO_TOP_BOTTOM_HALF = 6, STEREO_NONE = 7"))
    , _stereoStrategy(initData(&_stereoStrategy, "strategy","Stereo Strategy: PARALLEL = 0 OR TOEDIN = 1"))
    , _stereoShift(initData(&_stereoShift, "baseline","Stereoscopic Baseline"))
{
}

StereoCamera::~StereoCamera()
{
}

void StereoCamera::setCurrentSide(Side newSide)
{
    currentSide = newSide;
}

BaseCamera::Side StereoCamera::getCurrentSide()
{
    return currentSide;
}

void StereoCamera::setStereoEnabled(bool newEnable)
{
    _stereoEnabled.setValue(newEnable);
}

bool StereoCamera::getStereoEnabled()
{
    return _stereoEnabled.getValue();
}

void StereoCamera::setStereoMode(StereoMode newMode)
{
    _stereoMode.setValue(static_cast<int>(newMode));
}

BaseCamera::StereoMode StereoCamera::getStereoMode()
{
    return static_cast<StereoMode>(_stereoMode.getValue());
}

void StereoCamera::setStereoStrategy(StereoStrategy newStrategy)
{
    _stereoStrategy.setValue(static_cast<int>(newStrategy));
}

BaseCamera::StereoStrategy StereoCamera::getStereoStrategy()
{
    return static_cast<StereoStrategy>(_stereoStrategy.getValue());
}

void StereoCamera::setStereoShift(double newShift)
{
    _stereoShift.setValue(newShift);
}

double StereoCamera::getStereoShift()
{
    return _stereoShift.getValue();
}

} // namespace visual

} // namespace component

} // namespace sofa
