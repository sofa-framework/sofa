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
#include "VideoRecorderFFMpeg.h"
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace gui
{

namespace hRecorder
{

VideoRecorderFFmpeg::VideoRecorderFFmpeg(const int fps, const int width, const int height, const char* filename, const int codec_id)
    : fps(fps)
    , width(width)
    , height(height)
    , filename(filename)
    , codec_id(codec_id)
    , m_nFrames(0)
{
}

VideoRecorderFFmpeg::~VideoRecorderFFmpeg()
{
}

void VideoRecorderFFmpeg::start(void)
{
    AVCodec *codec;
    int ret;
    msg_info("VideoRecorder") << "Start recording ... " << filename;
    av_register_all();

    avformat_alloc_output_context2(&oc, NULL, "mp4", filename.c_str());
    if (!oc) {
        msg_error("VideoRecorder") << "Could not allocate format context.";
        exit(1);
    }

    if (oc->oformat->video_codec == AV_CODEC_ID_NONE) {
        msg_error("VideoRecorder") << "Could not allocate format context.";
        exit(1);
    }

    codec = avcodec_find_encoder(oc->oformat->video_codec);
    if (!codec) {
        msg_error("VideoRecorder") << "Codec not found";
        exit(1);
    }

    //st = avformat_new_stream(oc, NULL);
    st = avformat_new_stream(oc, codec);
    if (!st) {
        msg_error("VideoRecorder") << "Could not allocate stream";
        exit(1);
    }
    st->id = oc->nb_streams-1;

    AVCodec *pCodec = avcodec_find_decoder(st->codecpar->codec_id);
    enc = avcodec_alloc_context3(pCodec);

    if (!enc) {
        msg_error("VideoRecorder") << "Could not allocate video codec context";
        exit(1);
    }

    enc->codec_id = oc->oformat->video_codec;
    enc->bit_rate = 8000000; // maybe I need to adjust it
    enc->width = width;
    enc->height = height;
    st->time_base = (AVRational){1, fps};
    enc->time_base = st->time_base;
    //enc->framerate = (AVRational){fps, 1};
    enc->gop_size = 10;
    enc->max_b_frames = 1;
    enc->pix_fmt = AV_PIX_FMT_YUV420P;

    if (codec_id == AV_CODEC_ID_H264)
        av_opt_set(enc->priv_data, "preset", "slow", 0);

    /* open the codec */
    ret = avcodec_open2(enc, codec, NULL);
    if (ret < 0) {
        msg_error("VideoRecorder") << "Could not open codec";
        exit(1);
    }

    /* open the output file, if needed */
    if (!(oc->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&oc->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            msg_error("VideoRecorder") << "Could not open " << filename;
            exit(1);
        }
    }
    /* Write the stream header, if any. */
    ret = avformat_write_header(oc, NULL);
    if (ret < 0) {
        msg_error("VideoRecorder") << "Could not write header to output file : " << filename;
        exit(1);
    }

    m_frame = av_frame_alloc();
    if (!m_frame)
    {
        msg_error("VideoRecorder") << "Could not allocate video frame";
        exit(1);
    }
    m_frame->format = enc->pix_fmt;
    m_frame->width  = enc->width;
    m_frame->height = enc->height;

    ret = av_frame_get_buffer(m_frame, 32);
    if (ret < 0)
    {
        msg_error("VideoRecorder") << "Could not allocate the video frame data";
        exit(1);
    }
}

void VideoRecorderFFmpeg::encode(AVFrame* frame)
{
    int ret;
    //AVPacket pkt = { 0 };
    AVPacket pkt;
    av_init_packet(&pkt);
    if(frame)
    {
        frame->pts = av_rescale_q(m_nFrames, (AVRational){1, fps}, st->time_base);
        m_nFrames++;
    }

    ret = avcodec_send_frame(enc, frame);
    if (ret < 0) {
        msg_error("VideoRecorder") << "Error sending a frame for encoding";
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc, &pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            msg_error("VideoRecorder") << "Error during encoding";
            exit(1);
        }
        //av_packet_rescale_ts(pkt, *time_base, st->time_base);
        pkt.stream_index = st->index;
        av_interleaved_write_frame(oc, &pkt);
        av_packet_unref(&pkt);
    }
}

void VideoRecorderFFmpeg::encodeFrame(void)
{
    videoGLToFrame();
    int ret = av_frame_make_writable(m_frame);
    if (ret < 0)
    {
        msg_error("VideoRecorder") << "Frame is not writable";
        exit(1);
    }
    videoRGBToYUV();
    encode(m_frame);
}

void VideoRecorderFFmpeg::stop(void)
{
    // Free the frame and write remaining frames from the encoder
    av_frame_free(&m_frame);
    encode(nullptr);

    // Write the file trailer, if any
    av_write_trailer(oc);

    // Free the codec context
    avcodec_free_context(&enc);

    //sws_freeContext(sws_context);
    if (!(oc->oformat->flags & AVFMT_NOFILE))
    {
        avio_closep(&oc->pb);
    }

    avformat_free_context(oc);
}

void VideoRecorderFFmpeg::videoGLToFrame(void)
{
    int cur_gl, cur_rgb, nvals;
    const int format_nchannels = 4;
    nvals = format_nchannels * width * height;
    m_rgb = (uint8_t*)realloc(m_rgb, nvals * sizeof(uint8_t));
    GLubyte *pixels = (GLubyte*)malloc(nvals * sizeof(GLubyte));
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cur_gl  = format_nchannels * (width * (height - i - 1) + j);
            cur_rgb = format_nchannels * (width * i + j);
            for (int k = 0; k < format_nchannels; k++)
                (m_rgb)[cur_rgb + k] = (pixels)[cur_gl + k];
        }
    }
    free(pixels);
}

void VideoRecorderFFmpeg::videoRGBToYUV(void)
{
    const int in_linesize[1] = { 4 * enc->width };
    sws_context = sws_getCachedContext(sws_context,
                                       enc->width, enc->height, AV_PIX_FMT_RGBA,
                                       enc->width, enc->height, AV_PIX_FMT_YUV420P,
                                       0, NULL, NULL, NULL);

    sws_scale(sws_context, (const uint8_t * const *)&m_rgb, in_linesize, 0, enc->height, m_frame->data, m_frame->linesize);
}

} // namespace hRecorder

} // namespace gui

} // namespace sofa
