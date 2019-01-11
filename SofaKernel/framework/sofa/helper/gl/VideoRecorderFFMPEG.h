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
#ifndef SOFA_HELPER_GL_VIDEORECORDER_FFMPEG_H
#define SOFA_HELPER_GL_VIDEORECORDER_FFMPEG_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/helper.h>
#include <sofa/helper/gl/template.h>

#include <string>
#include <iostream>


namespace sofa
{

namespace helper
{

namespace gl
{

class SOFA_HELPER_API VideoRecorderFFMPEG
{
protected:
    

    unsigned int bitrate;
    uint8_t *videoOutbuf;
    int videoOutbufSize;

    
    FILE* pFile;
    int pWidth, pHeight;
    int pFrameCount;

    std::string p_filename;
    unsigned int p_framerate;

    std::string _prefix;
    int _counter;
    

    FILE* _ffmpeg;
    unsigned char* _buffer;

public:

    VideoRecorderFFMPEG();
    ~VideoRecorderFFMPEG();
    bool init(const std::string& filename, int width, int height, unsigned int framerate, unsigned int bitrate, const std::string& codec="");
   
    void addFrame();
    void saveVideo();
    void finishVideo();

    
    void setPrefix(const std::string v) { _prefix = v; }

    std::string findFilename(const unsigned int bitrate, const unsigned int framerate, std::string& extension);

protected:


};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif //SOFA_HELPER_GL_VIDEORECORDER_FFMPEG_H
