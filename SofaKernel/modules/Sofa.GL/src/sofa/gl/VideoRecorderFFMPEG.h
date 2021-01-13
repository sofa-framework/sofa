/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gl/config.h>
#include <sofa/gl/template.h>

#include <string>
#include <iostream>


namespace sofa::gl
{

class SOFA_GL_API VideoRecorderFFMPEG
{
protected:

    std::string m_ffmpegExecPath;
    int m_viewportWidth, m_viewportHeight;
    int m_ffmpegWidth, m_ffmpegHeight;
    int m_FrameCount;

    std::string m_filename;
    unsigned int m_framerate;

    std::string m_prefix;

    FILE* m_ffmpeg;

    unsigned char* m_viewportBuffer;
    size_t m_viewportBufferSize; // size in bytes

    unsigned char* m_ffmpegBuffer;
    size_t m_ffmpegBufferSize; // size in bytes

    int m_pixelFormatSize; // size in bytes

public:

    VideoRecorderFFMPEG();
    ~VideoRecorderFFMPEG();

    bool init(const std::string& ffmpeg_exec_filepath, const std::string& filename, int width, int height, unsigned int framerate, unsigned int bitrate, const std::string& codec="");

    void addFrame();
    void saveVideo();
    void finishVideo();


    void setPrefix(const std::string v) { m_prefix = v; }

    std::string findFilename(const unsigned int bitrate, const unsigned int framerate, const std::string& extension);

};

} // namespace sofa::gl
