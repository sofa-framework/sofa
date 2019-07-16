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
#include "GLBackend.h"

#include <sofa/helper/system/FileRepository.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

GLBackend::GLBackend()
    : m_texLogo(nullptr)
{
}

GLBackend::~GLBackend()
{
}

void GLBackend::setPickingMethod(sofa::gui::PickHandler* pick, sofa::component::configurationsetting::ViewerSetting* viewerConf)
{
    if ( viewerConf->objectPickingMethod.getValue().getSelectedId() == gui::PickHandler::RAY_CASTING)
        pick->setPickingMethod( gui::PickHandler::RAY_CASTING );
    else
        pick->setPickingMethod( gui::PickHandler::SELECTION_BUFFER);
}

void GLBackend::setPrefix(const std::string& prefix)
{
    m_capture.setPrefix(prefix);

#ifdef SOFA_HAVE_FFMPEG_EXEC
    m_videoRecorderFFMPEG.setPrefix(prefix);
#endif // SOFA_HAVE_FFMPEG_EXEC
}

const std::string GLBackend::screenshotName()
{
    return m_capture.findFilename().c_str();
}

void GLBackend::screenshot(const std::string& filename, int compression_level)
{
    m_capture.saveScreen(filename, compression_level);
}

void GLBackend::setBackgroundImage(helper::io::Image* image)
{
    if(m_texLogo)
    {
        delete m_texLogo;
        m_texLogo = nullptr;
    }

    m_texLogo = new helper::gl::Texture( image );
    m_texLogo->init();
}

bool GLBackend::initRecorder( int width, int height, unsigned int framerate, unsigned int bitrate, const std::string& codec)
{
    bool res = true;
#ifdef SOFA_HAVE_FFMPEG_EXEC
    std::string videoFilename = m_videoRecorderFFMPEG.findFilename(framerate, bitrate / 1024, codec);

    res = m_videoRecorderFFMPEG.init(videoFilename, width, height, framerate, bitrate, codec);
#endif // SOFA_HAVE_FFMPEG_EXEC

    return res;
}

void GLBackend::endRecorder()
{
#ifdef SOFA_HAVE_FFMPEG_EXEC
    m_videoRecorderFFMPEG.finishVideo();
#endif //SOFA_HAVE_FFMPEG_EXEC
}

void GLBackend::addFrameRecorder()
{
#ifdef SOFA_HAVE_FFMPEG_EXEC
    m_videoRecorderFFMPEG.addFrame();
#endif //SOFA_HAVE_FFMPEG_EXEC
}


}
}
}
}
