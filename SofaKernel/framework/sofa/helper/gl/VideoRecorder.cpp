/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
    img_convert_ctx = NULL;

    if (!VideoRecorder::FFMPEG_INITIALIZED)
    {
        av_register_all();

        VideoRecorder::FFMPEG_INITIALIZED = true;
    }
}

VideoRecorder::~VideoRecorder()
{
    if (pFormatContext)
        finishVideo();
}

AVFrame *VideoRecorder::alloc_picture(PixelFormat pix_fmt, int width, int height)
{
    AVFrame *picture;
    uint8_t *picture_buf;
    int size;

    picture = avcodec_alloc_frame();
    if (!picture)
        return NULL;
    size = avpicture_get_size(pix_fmt, width, height);
    picture_buf = (uint8_t *) av_malloc(size*2);
    if (!picture_buf)
    {
        av_free(picture);
        return NULL;
    }
    avpicture_fill((AVPicture *)picture, picture_buf,
            pix_fmt, width, height);
    return picture;
}

AVStream *VideoRecorder::add_video_stream(AVFormatContext *oc, AVCodecID codec_id, const std::string& codec)
{
    AVCodecContext *c;
    AVStream *st;

    st = av_new_stream(oc, 0);
    if (!st)
        return NULL;

    c = st->codec;
    c->codec_id = codec_id;
    c->codec_type = AVMEDIA_TYPE_VIDEO;

    /* put sample parameters */
    c->bit_rate = p_bitrate;
    /* resolution must be a multiple of two */
    c->width = pWidth;
    c->height = pHeight;
    /* time base: this is the fundamental unit of time (in seconds) in terms
       of which frame timestamps are represented. for fixed-fps content,
       timebase should be 1/framerate and timestamp increments should be
       identically 1. */
    c->time_base.num = 1;
    c->time_base.den = p_framerate;
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


    // override the codec or its parameters if requested
    if (!codec.empty())
    {
        if (codec == std::string("h264"))
        {
            c->codec_id = CODEC_ID_H264;
            // libx264-medium.ffpreset
            c->me_range = 16;
            c->max_qdiff = 4;
            c->qmin = 10;
            c->qmax = 51;
            c->qcompress = 0.6;
            c->max_b_frames= 3;
            c->me_cmp |= FF_CMP_CHROMA;
//            c->partitions |= X264_PART_I8X8 | X264_PART_I4X4 |
//                    X264_PART_P8X8 | X264_PART_B8X8;
            c->me_method = ME_HEX;

            //c->gop_size = 250;
            c->keyint_min = 10;

            c->scenechange_threshold = 40;
            c->i_quant_factor = 0.71;
            c->b_frame_strategy = 1;
//            c->directpred = 1;
            c->trellis = 1;
            c->flags2 = CODEC_FLAG2_FAST    | CODEC_FLAG2_NO_OUTPUT |
                    CODEC_FLAG2_LOCAL_HEADER | CODEC_FLAG2_IGNORE_CROP |  CODEC_FLAG2_CHUNKS;
    //        c->weighted_p_pred = 2;
        }
        else if (codec == std::string("mpeg4"))
        {
            c->codec_id = CODEC_ID_MPEG4;
        }
        else if (codec == std::string("mjpeg"))
        {
            c->codec_id = CODEC_ID_MJPEG;
        }
        else if (codec == std::string("lossless"))
        {
            c->codec_id = CODEC_ID_H264;
            // libx264-fast.ffpreset with qp=0
            c->me_range = 16;
            c->max_qdiff = 4;
            c->qmin = 0; //10;
            c->qmax = 0; //51;
            c->qcompress = 0.6;
            c->max_b_frames= 3;
            c->me_cmp |= FF_CMP_CHROMA;
//            c->partitions |= X264_PART_I8X8 | X264_PART_I4X4 |
 //                   X264_PART_P8X8 | X264_PART_B8X8;
            c->me_method = ME_HEX;

            //c->gop_size = 250;
            c->keyint_min = 10;

            c->scenechange_threshold = 40;
            c->i_quant_factor = 0.71;
            c->b_frame_strategy = 1;
       //     c->directpred = 1;
            c->trellis = 1;
			c->flags2 = CODEC_FLAG2_FAST    | CODEC_FLAG2_NO_OUTPUT |
			CODEC_FLAG2_LOCAL_HEADER | CODEC_FLAG2_IGNORE_CROP |  CODEC_FLAG2_CHUNKS;
    //        c->weighted_p_pred = 2;

//            c->cqp = 0;

            /*
                        c->me_range = 16;
                        c->max_qdiff = 4;
                        c->qmin = 10;
                        c->qmax = 51;
                        c->qcompress = 0.6;
                        //c->max_b_frames= 3;
                        c->me_cmp |= FF_CMP_CHROMA;
                        c->partitions |= X264_PART_I4X4 | X264_PART_P8X8;
                        c->partitions &= ~( X264_PART_I8X8 | X264_PART_I4X4 | X264_PART_B8X8 );
                        c->me_method = ME_HEX;

                        c->gop_size = 250;
                        c->keyint_min = 25;

                        c->scenechange_threshold = 40;
                        c->i_quant_factor = 0.71;
                        c->b_frame_strategy = 1;
                        c->directpred = 1;
                        c->flags2 |= CODEC_FLAG2_FASTPSKIP;
                        c->weighted_p_pred = 0;
            */
        }
    }

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
    if (avcodec_open2(c, codec,NULL) < 0)
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

void VideoRecorder::fill_image(AVFrame *pict, int /*frame_index*/, int width, int height)
{
    glReadBuffer(GL_FRONT);
    // Read bits from color buffer
    uint8_t *avbuf = pict->data[0];
    unsigned int lsize = pict->linesize[0];
    uint8_t *glbuf = avbuf+lsize*height;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glReadPixels(0, 0, pWidth, pHeight, GL_RGB, GL_UNSIGNED_BYTE, glbuf);
    for(int y=0; y<height; y++)
    {
        memcpy(avbuf+(y*lsize), glbuf+((height-1-y)*width*3), width*3);
    }
}


bool VideoRecorder::write_delayed_video_frame(AVFormatContext *oc, AVStream *st)
{
    if (oc->oformat->flags & AVFMT_RAWPICTURE)
        return true;

    int out_size = 1, ret = 0;
    AVCodecContext *c;

    c = st->codec;

    while (out_size > 0)
    {
        out_size = avcodec_encode_video(c, videoOutbuf, videoOutbufSize, NULL);
        if (out_size > 0)
        {
            AVPacket pkt;
            av_init_packet(&pkt);

            pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, st->time_base);
            if(c->coded_frame->key_frame)
                pkt.flags |= AV_PKT_FLAG_KEY;
            pkt.stream_index= st->index;
            pkt.data= videoOutbuf;
            pkt.size= out_size;

            /* write the compressed frame in the media file */
            ret = av_interleaved_write_frame(oc, &pkt);
        }
    }
    if (ret != 0)
    {
        std::cerr << "Error while writing video frame"<<std::endl;
        return false;
    }

    return true;
}
bool VideoRecorder::write_video_frame(AVFormatContext *oc, AVStream *st)
{
    int sws_flags = SWS_BICUBIC;

    int out_size, ret;
    AVCodecContext *c;

    c = st->codec;
    {
        if (c->pix_fmt != SOFA_GL_PIX_FMT)
        {
            // as we only generate a SOFA_GL_PIX_FMT picture, we must convert it
            //   to the codec pixel format if needed
            if (img_convert_ctx == NULL)
            {
                std::cout << "Initialize image conversion context"<<std::endl;
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

        pkt.flags |= AV_PKT_FLAG_KEY;
        pkt.stream_index= st->index;
        pkt.data= (uint8_t *)pPicture;
        pkt.size= sizeof(AVPicture);

        ret = av_interleaved_write_frame(oc, &pkt);
    }
    else
    {
        /* encode the image */
        pPicture->pts = pFrameCount;
        out_size = avcodec_encode_video(c, videoOutbuf, videoOutbufSize, pPicture);
        /* if zero size, it means the image was buffered */
        if (out_size > 0)
        {
            AVPacket pkt;
            av_init_packet(&pkt);
            std::cout << "Encoded Video Frame: " << c->coded_frame->pts << std::endl;
            pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, st->time_base);
            if(c->coded_frame->key_frame)
                pkt.flags |= AV_PKT_FLAG_KEY;
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
    if (img_convert_ctx)
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = NULL;
    }

}

///// Public methods


bool VideoRecorder::init(const std::string& filename, unsigned int framerate, unsigned int bitrate, const std::string& codec )
{
    std::cout << "START recording to " << filename << " ( ";
    if (!codec.empty()) std::cout << codec << ", ";
    std::cout << framerate << " FPS, " << bitrate << " b/s";
    std::cout << " )" << std::endl;
    //std::string filename = findFilename();
    p_filename = filename;
    p_framerate = framerate;
    p_bitrate = bitrate;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    pWidth = viewport[2];
    pHeight= viewport[3];

    pFrameCount = 0;

    if (img_convert_ctx)
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = NULL;
    }

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(52,64,0)
    pFormat = av_guess_format(NULL, filename.c_str(), NULL);
#else
    pFormat = guess_format(NULL, filename.c_str(), NULL);
#endif

    if (!pFormat)
    {
        std::cerr << "Could not deduce output format from file extension: using MPEG" << std::endl;
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(52,64,0)
        pFormat = av_guess_format("mpeg", NULL, NULL);
#else
        pFormat = guess_format("mpeg", NULL, NULL);
#endif
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

    // add the audio and video streams and initialize the codecs
    pVideoStream = NULL;


    if (pFormat->video_codec != CODEC_ID_NONE)
    {
        pVideoStream = add_video_stream(pFormatContext, pFormat->video_codec, codec);
    }

    // set the output parameters (must be done even if no parameters)
/*    if (av_set_parameters(pFormatContext, NULL) < 0)
    {
        std::cerr <<"Invalid output format parameters" << std::endl;
        return false;
    }*/

    // now that all the parameters are set, we can open the audio and
    //   video codecs and allocate the necessary encode buffers */
    if (pVideoStream)
        open_video(pFormatContext, pVideoStream);

    // open the output file, if needed
    if (!(pFormat->flags & AVFMT_NOFILE))
    {
        if (avio_open(&pFormatContext->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            std::cerr <<"Could not open '%s'" << std::endl;
            return false;
        }
    }

    // write the stream header, if any
    avformat_write_header(pFormatContext,NULL);

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
    write_delayed_video_frame(pFormatContext, pVideoStream);
    // write the trailer, if any.  the trailer must be written
    // before you close the CodecContexts open when you wrote the
    // header; otherwise write_trailer may try to use memory that
    // was freed on av_codec_close()
    av_write_trailer(pFormatContext);

    //Dump to stdout
    av_dump_format(pFormatContext, 0, p_filename.c_str(), 1);

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
        avio_close(pFormatContext->pb);
    }

    // free the stream
    av_free(pFormatContext);
    pFormatContext = NULL;

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

