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
#include <sofa/helper/gl/VideoRecorder.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <cstdio>		// sprintf and friends

namespace sofa
{

namespace helper
{

namespace gl
{

//http://wiki.aasimon.org/doku.php?id=ffmpeg:output_example&s=avcodec%20encode%20video

bool VideoRecorder::FFMPEG_INITIALIZED = false;

#define STREAM_PIX_FMT PIX_FMT_YUV420P
#define SOFA_GL_PIX_FMT PIX_FMT_RGB24


VideoRecorder::VideoRecorder()
    : prefix("screencast"), counter(-1)
    , pFormatContext(NULL), pCodecContext(NULL), pCodec(NULL)
    , p_framerate(25), p_bitrate(400000)
{
    if (!VideoRecorder::FFMPEG_INITIALIZED)
    {
        av_register_all();

        VideoRecorder::FFMPEG_INITIALIZED = true;
    }
}

AVFrame *VideoRecorder::alloc_picture(int pix_fmt, int width, int height)
{
    AVFrame *picture;
    uint8_t *picture_buf;
    int size;

    picture = avcodec_alloc_frame();
    if (!picture)
        return NULL;
    size = avpicture_get_size(pix_fmt, width, height);
    picture_buf = (uint8_t *) av_malloc(size);
    if (!picture_buf)
    {
        av_free(picture);
        return NULL;
    }
    avpicture_fill((AVPicture *)picture, picture_buf,
            pix_fmt, width, height);
    return picture;
}

AVStream *VideoRecorder::add_video_stream(AVFormatContext *oc, int codec_id)
{
    AVCodecContext *c;
    AVStream *st;

    st = av_new_stream(oc, 0);
    if (!st)
        return NULL;

    c = st->codec;
    c->codec_id = (CodecID) codec_id;
    c->codec_type = CODEC_TYPE_VIDEO;

    /* put sample parameters */
    c->bit_rate = p_bitrate;
    /* resolution must be a multiple of two */
    c->width = pWidth;
    c->height = pHeight;
    /* time base: this is the fundamental unit of time (in seconds) in terms
       of which frame timestamps are represented. for fixed-fps content,
       timebase should be 1/framerate and timestamp increments should be
       identically 1. */
    c->time_base.den = p_framerate;
    c->time_base.num = 1;
    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = STREAM_PIX_FMT;
    if (c->codec_id == CODEC_ID_MPEG2VIDEO)
    {
        /* just for testing, we also add B frames */
        c->max_b_frames = 2;
    }
    if (c->codec_id == CODEC_ID_MPEG1VIDEO)
    {
        /* Needed to avoid using macroblocks in which some coeffs overflow.
           This does not happen with normal video, it just happens here as
           the motion of the chroma plane does not match the luma plane. */
        c->mb_decision=2;
    }
    // some formats want stream headers to be separate
    if(oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

    return st;
}

bool VideoRecorder::open_video(AVFormatContext *oc, AVStream *st)
{
    AVCodec *codec;
    AVCodecContext *c;

    c = st->codec;

    /* find the video encoder */
    codec = avcodec_find_encoder(c->codec_id);
    if (!codec)
    {
        std::cerr <<"codec not found"<<std::endl;
        return false;
    }

    /* open the codec */
    if (avcodec_open(c, codec) < 0)
    {
        std::cerr <<"could not open codec"<<std::endl;
        return false;
    }

    videoOutbuf = NULL;
    if (!(oc->oformat->flags & AVFMT_RAWPICTURE))
    {
        // allocate output buffer
        // XXX: API change will be done
        // buffers passed into lav* can be allocated any way you prefer,
        //   as long as they're aligned enough for the architecture, and
        //   they're freed appropriately (such as using av_free for buffers
        //   allocated with av_malloc)
        videoOutbufSize = 200000;
        videoOutbuf = (uint8_t*) av_malloc(videoOutbufSize);
    }

    // allocate the encoded raw picture
    pPicture = alloc_picture(c->pix_fmt, c->width, c->height);
    if (!pPicture)
    {
        std::cerr <<"Could not allocate picture"<<std::endl;
        return false;
    }

    // if the output format is not YUV420P, then a temporary YUV420P
    //   picture is needed too. It is then converted to the required
    //   output format
    pTempPicture = NULL;
    if (c->pix_fmt != SOFA_GL_PIX_FMT)
    {
        pTempPicture = alloc_picture(SOFA_GL_PIX_FMT, c->width, c->height);
        if (!pTempPicture)
        {
            std::cerr <<"Could not allocate temporary picture"<<std::endl;
            return false;
        }
    }
    return true;
}

void VideoRecorder::fill_image(AVFrame *pict, int frame_index, int width, int height)
{
    int i;

    io::ImageBMP img;

    img.init(pWidth, pHeight, 1, 1, io::Image::UNORM8, io::Image::RGB);
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, pWidth, pHeight, GL_RGB, GL_UNSIGNED_BYTE, img.getPixels());

    i = frame_index;

    if (SOFA_GL_PIX_FMT == PIX_FMT_RGB24)
    {
        for(int y=0; y<height; y++)
        {
            memcpy(&pict->data[0][y * pict->linesize[0]], img.getPixels()+(height-y)*width*3, width*3);
        }

    }

}


bool VideoRecorder::write_video_frame(AVFormatContext *oc, AVStream *st)
{
    int sws_flags = SWS_BICUBIC;

    int out_size, ret;
    AVCodecContext *c;
    struct SwsContext *img_convert_ctx = NULL;

    c = st->codec;
    /* TODO ...
        if (pFrameCount >= STREAM_NB_FRAMES)
        {
            // no more frame to compress. The codec has a latency of a few
            //   frames if using B frames, so we get the last frames by
            //   passing the same picture again
        }
        else*/
    {
        if (c->pix_fmt != SOFA_GL_PIX_FMT)
        {
            // as we only generate a SOFA_GL_PIX_FMT picture, we must convert it
            //   to the codec pixel format if needed
            if (img_convert_ctx == NULL)
            {
                img_convert_ctx = sws_getContext(c->width, c->height,
                        SOFA_GL_PIX_FMT,
                        c->width, c->height,
                        c->pix_fmt,
                        sws_flags, NULL, NULL, NULL);

                if (img_convert_ctx == NULL)
                {
                    std::cerr << "Cannot initialize the conversion context"<<std::endl;
                    return false;
                }
            }
            fill_image(pTempPicture, pFrameCount, c->width, c->height);

            sws_scale(img_convert_ctx, pTempPicture->data, pTempPicture->linesize,
                    0, c->height, pPicture->data, pPicture->linesize);
        }
        else
        {
            fill_image(pPicture, pFrameCount, c->width, c->height);
        }
    }


    if (oc->oformat->flags & AVFMT_RAWPICTURE)
    {
        /* raw video case. The API will change slightly in the near
           futur for that */
        AVPacket pkt;
        av_init_packet(&pkt);

        pkt.flags |= PKT_FLAG_KEY;
        pkt.stream_index= st->index;
        pkt.data= (uint8_t *)pPicture;
        pkt.size= sizeof(AVPicture);

        ret = av_interleaved_write_frame(oc, &pkt);
    }
    else
    {
        /* encode the image */
        out_size = avcodec_encode_video(c, videoOutbuf, videoOutbufSize, pPicture);
        /* if zero size, it means the image was buffered */
        if (out_size > 0)
        {
            AVPacket pkt;
            av_init_packet(&pkt);

            //WTF? if (c->coded_frame->pts != AV_NOPTS_VALUE)
            pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, st->time_base);
            if(c->coded_frame->key_frame)
                pkt.flags |= PKT_FLAG_KEY;
            pkt.stream_index= st->index;
            pkt.data= videoOutbuf;
            pkt.size= out_size;

            /* write the compressed frame in the media file */
            ret = av_interleaved_write_frame(oc, &pkt);
        }
        else
        {
            ret = 0;
        }
    }
    if (ret != 0)
    {
        std::cerr << "Error while writing video frame"<<std::endl;
        return false;
    }
    pFrameCount++;

    return true;
}

void VideoRecorder::close_video(AVFormatContext * /*oc */, AVStream *st)
{
    avcodec_close(st->codec);
    av_free(pPicture->data[0]);
    av_free(pPicture);
    if (pTempPicture)
    {
        av_free(pTempPicture->data[0]);
        av_free(pTempPicture);
    }
    av_free((uint8_t*)videoOutbuf);

}

///// Public methods

bool VideoRecorder::init(const std::string& filename, unsigned int framerate, unsigned int bitrate )
{
    //std::string filename = findFilename();
    p_filename = filename;
    p_framerate = framerate;
    p_bitrate = bitrate;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    pWidth = viewport[2];
    pHeight= viewport[3];

    pFormat = guess_format(NULL, filename.c_str(), NULL);

    if (!pFormat)
    {
        std::cerr << "Could not deduce output format from file extension: using MPEG" << std::endl;
        pFormat = guess_format("mpeg", NULL, NULL);
    }
    if (!pFormat)
    {
        std::cerr <<"Could not find suitable output format" << std::endl;
        return false;
    }

    pFormatContext = avformat_alloc_context();
    if (!pFormatContext)
    {
        std::cerr <<"Error format context" << std::endl;
        return false;
    }

    // allocate the output media context
    pFormatContext->oformat = pFormat;
    snprintf(pFormatContext->filename, sizeof(pFormatContext->filename), "%s", filename.c_str());

    // add the audio and video streams using the default format codecs
    // and initialize the codecs
    pVideoStream = NULL;

    if (pFormat->video_codec != CODEC_ID_NONE)
    {
        pVideoStream = add_video_stream(pFormatContext, pFormat->video_codec);
    }

    // set the output parameters (must be done even if no parameters)
    if (av_set_parameters(pFormatContext, NULL) < 0)
    {
        std::cerr <<"Invalid output format parameters" << std::endl;
        return false;
    }

    // now that all the parameters are set, we can open the audio and
    //   video codecs and allocate the necessary encode buffers */
    if (pVideoStream)
        open_video(pFormatContext, pVideoStream);

    // open the output file, if needed
    if (!(pFormat->flags & AVFMT_NOFILE))
    {
        if (url_fopen(&pFormatContext->pb, filename.c_str(), URL_WRONLY) < 0)
        {
            std::cerr <<"Could not open '%s'" << std::endl;
            return false;
        }
    }

    // write the stream header, if any
    av_write_header(pFormatContext);

    std::cout << "Start Recording in " << filename << std::endl;
    //everything is ok
    return true;
}

void VideoRecorder::addFrame()
{
    write_video_frame(pFormatContext, pVideoStream);
}

void VideoRecorder::finishVideo()
{
    // write the trailer, if any.  the trailer must be written
    // before you close the CodecContexts open when you wrote the
    // header; otherwise write_trailer may try to use memory that
    // was freed on av_codec_close()
    av_write_trailer(pFormatContext);

    //Dump to stdout
    dump_format(pFormatContext, 0, p_filename.c_str(), 1);

    // close each codec
    if (pVideoStream)
        close_video(pFormatContext, pVideoStream);

    // free the streams
    for(unsigned int i = 0; i < pFormatContext->nb_streams; i++)
    {
        av_freep(&pFormatContext->streams[i]->codec);
        av_freep(&pFormatContext->streams[i]);
    }

    if (!(pFormat->flags & AVFMT_NOFILE))
    {
        // close the output file
        url_fclose(pFormatContext->pb);
    }

    // free the stream
    av_free(pFormatContext);

    std::cout << p_filename << " written" << std::endl;
}

std::string VideoRecorder::findFilename(const std::string& extension)
{
    std::string filename;
    char buf[32];
    int c;
    c = 0;
    struct stat st;
    do
    {
        ++c;
        sprintf(buf, "%04d",c);
        filename = prefix;
        filename += buf;
        filename += ".";
        filename += extension;
    }
    while (stat(filename.c_str(),&st)==0);
    counter = c+1;

    sprintf(buf, "%04d",c);
    filename = prefix;
    filename += buf;
    filename += ".";
    filename += extension;

    return filename;
}


} // namespace gl

} // namespace helper

} // namespace sofa

