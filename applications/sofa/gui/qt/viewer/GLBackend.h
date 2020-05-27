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
#ifndef SOFA_GLBACKEND_H
#define SOFA_GLBACKEND_H

#include <sofa/gui/qt/SofaGuiQt.h>

#include <sofa/gui/qt/viewer/EngineBackend.h>
#include <sofa/gui/PickHandler.h>
#include <SofaGraphComponent/ViewerSetting.h>

#include <sofa/helper/gl/Capture.h>
#include <sofa/helper/gl/Texture.h>

#if SOFAGUIQT_HAVE_FFMPEG_EXEC
#include <sofa/helper/gl/VideoRecorderFFMPEG.h>
#endif // SOFAGUIQT_HAVE_FFMPEG_EXEC


namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

class SOFA_SOFAGUIQT_API GLBackend : public EngineBackend
{
public:
    GLBackend();
    virtual ~GLBackend();

    void setPickingMethod(sofa::gui::PickHandler* pick, sofa::component::configurationsetting::ViewerSetting* viewerConf);
    void setPrefix(const std::string& prefix);
    const std::string screenshotName();
    void screenshot(const std::string& filename, int compression_level = -1);
    void setBackgroundImage(helper::io::Image* image);
    void drawBackgroundImage(const int screenWidth, const int screenHeight);

    bool initRecorder(int width, int height, unsigned int framerate, unsigned int bitrate,const std::string& codecExtension="",  const std::string& codecName="");
    void endRecorder();
    void addFrameRecorder();

private:
    sofa::helper::gl::Capture m_capture;
    sofa::helper::gl::Texture* m_texLogo;
#if SOFAGUIQT_HAVE_FFMPEG_EXEC
    sofa::helper::gl::VideoRecorderFFMPEG m_videoRecorderFFMPEG;
#endif // SOFAGUIQT_HAVE_FFMPEG_EXEC

};

} // namespace viewer

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GLBACKEND_H
