/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_BASEGLVIEWER_H
#define SOFA_GUI_BASEGLVIEWER_H

#include "sofa/config.h"

#include "BaseViewer.h"





namespace sofa
{

namespace gui
{

class SOFA_SOFAGUI_API BaseGLViewer : public BaseViewer
{

public:
    BaseGLViewer();
    virtual ~BaseGLViewer() override;

    //Allow to configure your viewer using the Sofa Component, ViewerSetting
    virtual void configure(sofa::component::configurationsetting::ViewerSetting* viewerConf) override;

    //Fonctions needed to take a screenshot
    const std::string screenshotName() override;
    void setPrefix(const std::string& prefix, bool prependDirectory = true) override;
    virtual void screenshot(const std::string& filename, int compression_level =-1) override;

    virtual void setBackgroundImage(std::string imageFileName = std::string("textures/SOFA_logo.bmp")) override;
protected:
    sofa::helper::gl::Capture m_capture;
    sofa::helper::gl::Texture* m_texLogo;

#ifdef SOFA_HAVE_FFMPEG_EXEC
    sofa::helper::gl::VideoRecorderFFMPEG m_videoRecorderFFMPEG;
#endif // SOFA_HAVE_FFMPEG_EXEC
};

}
}

#endif // SOFA_GUI_BASEGLVIEWER_H
