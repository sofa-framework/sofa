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
#include <sofa/helper/gl/VideoRecorderFFMPEG.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>      //popen
#include <cstdio>		// sprintf and friends
#include <sstream>

namespace sofa
{

namespace helper
{

namespace gl
{

VideoRecorderFFMPEG::VideoRecorderFFMPEG()
    : m_framerate(25)
    , m_prefix("sofa_video")
    , m_counter(-1)
    , m_ffmpeg(nullptr)
    , m_buffer(nullptr)
{

}

VideoRecorderFFMPEG::~VideoRecorderFFMPEG()
{

}


bool VideoRecorderFFMPEG::init(const std::string& filename, int width, int height, unsigned int framerate, unsigned int bitrate, const std::string& codec )
{
    std::cout << "START recording to " << filename << " ( ";
    if (!codec.empty()) std::cout << codec << ", ";
    std::cout << framerate << " FPS, " << bitrate << " b/s";
    std::cout << " )" << std::endl;
    //std::string filename = findFilename();
    m_filename = filename;
    m_framerate = framerate;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    m_Width = width;// viewport[2];
    m_Height = height;// viewport[3];

    m_FrameCount = 0;

    m_buffer = new unsigned char [4*m_Width*m_Height];

    std::stringstream ss;
    ss << FFMPEG_EXEC_FILE << " -r " << m_framerate
        << " -f rawvideo -pix_fmt rgba "
        << " -s " << m_Width << "x" << m_Height
        << " -i - -threads 0  -y"
        << " -preset fast "
        << " -pix_fmt " << codec // yuv420p " // " yuv444p "
        << " -crf 17 "
        << " -vf vflip "
        << m_filename;
    
    const std::string& tmp = ss.str();

#ifdef WIN32
    m_ffmpeg = _popen(tmp.c_str(), "wb");
#else
    m_ffmpeg = popen(tmp.c_str(), "b");
#endif

    std::cout << "Start Recording in " << filename << std::endl;
    return true;
}

void VideoRecorderFFMPEG::addFrame()
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
   
    if ((viewport[2] != m_Width) || (viewport[3] != m_Height))
    {
        std::cout << "WARNING viewport changed during video capture from " << m_Width << "x" << m_Height << "  to  " << viewport[2] << "x" << viewport[3] << std::endl;
    }

    //glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGBA, GL_UNSIGNED_BYTE, m_buffer);

    glReadPixels(0, 0, m_Width, m_Height, GL_RGBA, GL_UNSIGNED_BYTE, (void*)m_buffer);

    fwrite(m_buffer, sizeof(unsigned char)*4*m_Width*m_Height, 1, m_ffmpeg);
    
    return;
}

void VideoRecorderFFMPEG::finishVideo()
{    
#ifdef WIN32
    _pclose(m_ffmpeg);
#else
    pclose(_ffmpeg);
#endif
    
    delete m_buffer;
    std::cout << m_filename << " written" << std::endl;
}

std::string VideoRecorderFFMPEG::findFilename(const unsigned int framerate, const unsigned int bitrate, const std::string& extension)
{
    std::string filename;
    char buf[32];
    int c;
    c = 0;
    struct stat st;
    do
    {
        ++c;
        sprintf(buf, "%04d", c);
        filename = m_prefix;
        filename += "_r" + std::to_string(framerate) + "_";
        //filename += +"_b" + std::to_string(bitrate) + "_";
        filename += buf;
        filename += ".";
        filename += extension;
    } while (stat(filename.c_str(), &st) == 0);
    m_counter = c + 1;

    sprintf(buf, "%04d", c);
    filename = m_prefix;
    filename += "_r" + std::to_string(framerate) + "_";
    //filename += +"_b" + std::to_string(bitrate) + "_";
    filename += buf;
    filename += ".";
    filename += extension;

    return filename;
}

} // namespace gl

} // namespace helper

} // namespace sofa

