/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_GL_VIDEORECORDER_H
#define SOFA_HELPER_GL_VIDEORECORDER_H

#include <sofa/helper/helper.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/ImageBMP.h>

#include <string>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace sofa
{

namespace helper
{

namespace gl
{

//using namespace sofa::defaulttype;

class SOFA_HELPER_API VideoRecorder
{
protected:
    std::string prefix;
    int counter;
    static bool FFMPEG_INITIALIZED;

    unsigned int bitrate;
    uint8_t *videoOutbuf;
    int videoOutbufSize;

    AVFormatContext *pFormatContext;
    AVOutputFormat *pFormat;
    AVCodecContext *pCodecContext;
    AVCodec *pCodec;
    AVFrame *pPicture,*pTempPicture;
    AVStream *pVideoStream;
    FILE* pFile;
    int pWidth, pHeight;
    int pFrameCount;

    std::string p_filename;
    unsigned int p_framerate, p_bitrate;

public:

    VideoRecorder();
    bool init(const std::string& filename, unsigned int framerate, unsigned int bitrate );
    void addFrame();
    void saveVideo();
    void finishVideo();

    void setPrefix(const std::string v) { prefix=v; }
    std::string findFilename(const std::string &v);

protected:
    AVStream *add_video_stream(AVFormatContext *oc, int codec_id);
    bool open_video(AVFormatContext *oc, AVStream *st);
    AVFrame *alloc_picture(int pix_fmt, int width, int height);
    bool write_video_frame(AVFormatContext *oc, AVStream *st);
    void fill_image(AVFrame *pict, int frame_index, int width, int height);
    void close_video(AVFormatContext *oc, AVStream *st);

};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif //SOFA_HELPER_GL_VIDEORECORDER_H
