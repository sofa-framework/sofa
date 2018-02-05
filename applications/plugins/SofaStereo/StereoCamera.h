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
    Data< bool > _stereoEnabled;
    Data< int > _stereoMode;
    Data< int > _stereoStrategy;
    Data< double > _stereoShift;
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
