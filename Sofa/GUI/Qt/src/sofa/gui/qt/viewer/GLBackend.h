/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gui/qt/config.h>

#include <sofa/gui/qt/viewer/EngineBackend.h>
#include <sofa/gui/common/PickHandler.h>
#include <sofa/component/setting/ViewerSetting.h>

#include <sofa/gl/Capture.h>
#include <sofa/gl/Texture.h>
#include <sofa/gl/VideoRecorderFFMPEG.h>


namespace sofa::gui::qt::viewer
{

class SOFA_GUI_QT_API GLBackend : public EngineBackend
{
public:
    GLBackend();
    virtual ~GLBackend();

    void setPickingMethod(sofa::gui::common::PickHandler* pick, sofa::component::setting::ViewerSetting* viewerConf);
    void setPrefix(const std::string& prefix);
    const std::string screenshotName();
    void screenshot(const std::string& filename, int compression_level = -1);
    void setBackgroundImage(helper::io::Image* image);
    void drawBackgroundImage(const int screenWidth, const int screenHeight);

    bool initRecorder(int width, int height, unsigned int framerate, unsigned int bitrate,const std::string& codecExtension="",  const std::string& codecName="");
    void endRecorder();
    void addFrameRecorder();

private:
    gl::Capture m_capture;
    gl::Texture* m_texLogo;
    gl::VideoRecorderFFMPEG m_videoRecorderFFMPEG;
};

} // namespace sofa::gui::qt::viewer
