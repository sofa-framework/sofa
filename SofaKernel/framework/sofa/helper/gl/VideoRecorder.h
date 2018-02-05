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
#ifndef SOFA_HELPER_GL_VIDEORECORDER_H
#define SOFA_HELPER_GL_VIDEORECORDER_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/helper.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/ImageBMP.h>

#include <string>
#include <iostream>

// FIX compilation issue (see http://code.google.com/p/ffmpegsource/issues/detail?id=11)
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
#include <stdint.h>

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
    struct SwsContext *img_convert_ctx;

    std::string p_filename;
    unsigned int p_framerate, p_bitrate;

public:

    VideoRecorder();
    ~VideoRecorder();
    bool init(const std::string& filename, unsigned int framerate, unsigned int bitrate, const std::string& codec="");
    void addFrame();
    void saveVideo();
    void finishVideo();

    void setPrefix(const std::string v) { prefix=v; }
    std::string findFilename(const std::string &v);

protected:
    AVStream *add_video_stream(AVFormatContext *oc, AVCodecID codec_id, const std::string& codec="");
    bool open_video(AVFormatContext *oc, AVStream *st);
    AVFrame *alloc_picture(PixelFormat pix_fmt, int width, int height);
    bool write_video_frame(AVFormatContext *oc, AVStream *st);
    bool write_delayed_video_frame(AVFormatContext *oc, AVStream *st);
    void fill_image(AVFrame *pict, int frame_index, int width, int height);
    void close_video(AVFormatContext *oc, AVStream *st);

};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif //SOFA_HELPER_GL_VIDEORECORDER_H
