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
#ifndef SOFA_STEREO_PLUGIN_STEREOCAMERA_H
#define SOFA_STEREO_PLUGIN_STEREOCAMERA_H

#include <sofa/core/visual/VisualModel.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <SofaBaseVisual/BaseCamera.h>
#include <SofaBaseVisual/InteractiveCamera.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class StereoCamera : public sofa::component::visualmodel::InteractiveCamera
{
public:
    SOFA_CLASS(StereoCamera, sofa::component::visualmodel::InteractiveCamera);

protected:
    Data< bool > _stereoEnabled; ///< Is the stereo mode initially enabled?
    Data< int > _stereoMode; ///< Stereo Mode: STEREO_AUTO = 0, STEREO_INTERLACED = 1, STEREO_FRAME_PACKING = 2, STEREO_SIDE_BY_SIDE = 3, STEREO_TOP_BOTTOM = 4, STEREO_SIDE_BY_SIDE_HALF = 5, STEREO_TOP_BOTTOM_HALF = 6, STEREO_NONE = 7
    Data< int > _stereoStrategy; ///< Stereo Strategy: PARALLEL = 0 OR TOEDIN = 1
    Data< double > _stereoShift; ///< Stereoscopic Baseline
    sofa::component::visualmodel::BaseCamera::Side currentSide;
    StereoCamera();
    virtual ~StereoCamera();

public:
    bool isStereo()
    {
        return true;
    }
    void setCurrentSide(Side newSide);
    Side getCurrentSide();
    void setStereoEnabled(bool newEnable);
    bool getStereoEnabled();
    void setStereoMode(StereoMode newMode);
    StereoMode getStereoMode();
    void setStereoStrategy(StereoStrategy newStrategy);
    StereoStrategy getStereoStrategy();
    void setStereoShift(double newShift);
    double getStereoShift();
};

} // namespace visual

} // namespace component

} // namespace sofa

#endif // SOFA_STEREO_PLUGIN_CAMERA_H
