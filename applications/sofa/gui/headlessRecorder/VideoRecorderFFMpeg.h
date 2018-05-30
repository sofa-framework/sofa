/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef VIDEORECORDERFFMPEG_H
#define VIDEORECORDERFFMPEG_H

#include <string>

// LIBAV
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libavformat/avformat.h>
}

// OPENGL
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glx.h>

namespace sofa
{

namespace gui
{

namespace hRecorder
{

class VideoRecorderFFmpeg
{
public:
    VideoRecorderFFmpeg(const int fps, const int width, const int height, const char *filename, const int codec_id);
    ~VideoRecorderFFmpeg();

    void start(void);
    void stop(void);
    void encodeFrame(void);

private:
    void videoRGBToYUV(void);
    void videoGLToFrame(void);
    void encode(AVFrame* frame);

    int fps;
    int width;
    int height;
    std::string filename;
    int codec_id;

    int m_nFrames;
    AVStream *st;
    AVFormatContext *oc;
    AVCodecContext *enc = NULL;
    AVFrame *m_frame;
    struct SwsContext *sws_context = NULL;
    uint8_t *m_rgb = NULL;


};

} // namespace hRecorder

} // namespace gui

} // namespace sofa

#endif // VIDEORECORDERFFMPEG_H
