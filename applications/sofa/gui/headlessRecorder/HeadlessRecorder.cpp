/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "HeadlessRecorder.h"

namespace sofa
{

namespace gui
{

namespace hRecorder
{

int HeadlessRecorder::width = 1920;
int HeadlessRecorder::height = 1080;
int HeadlessRecorder::recordTimeInSeconds = 5;
int HeadlessRecorder::fps = 60;
std::string HeadlessRecorder::fileName = "tmp";
bool HeadlessRecorder::saveAsVideo = false;
bool HeadlessRecorder::saveAsScreenShot = false;
bool HeadlessRecorder::recordUntilStopAnimate = false;

std::string HeadlessRecorder::recordTypeRaw = "wallclocktime";
RecordMode HeadlessRecorder::recordType = RecordMode::wallclocktime;
float HeadlessRecorder::skipTime = 0;

using namespace sofa::defaulttype;
using sofa::simulation::getSimulation;

SOFA_DECL_CLASS(HeadlessRecorder)
static sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
static glXMakeContextCurrentARBProc glXMakeContextCurrentARB = 0;

void static_handler(int /*signum*/)
{
    HeadlessRecorder::recordUntilStopAnimate = false;
    HeadlessRecorder::recordTimeInSeconds = 0;
}

// Class
HeadlessRecorder::HeadlessRecorder()
{
    groot = NULL;
    m_nFrames = 0;
    initVideoRecorder = true;
    initTexturesDone = false;
    vparams = core::visual::VisualParams::defaultInstance();
    vparams->drawTool() = &drawTool;

    signal(SIGTERM, static_handler);
    signal(SIGINT, static_handler);
}

HeadlessRecorder::~HeadlessRecorder()
{
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo_color);
    glDeleteRenderbuffers(1, &rbo_depth);
}

void HeadlessRecorder::parseRecordingModeOption()
{
    if (recordTypeRaw == "wallclocktime")
    {
        recordType = RecordMode::wallclocktime;
        skipTime = 0;
    }
    else if (recordTypeRaw == "simulationtime")
    {
        recordType = RecordMode::simulationtime;
        skipTime = 1.0/fps;
    }
    else
    {
        recordType = RecordMode::timeinterval;
        skipTime = std::stof(recordTypeRaw);
   }
}

int HeadlessRecorder::RegisterGUIParameters(ArgumentParser* argumentParser)
{
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%F-%X");

    argumentParser->addArgument(po::value<bool>(&saveAsScreenShot)->default_value(false)->implicit_value(true),         "picture", "enable picture mode (save as png)");
    argumentParser->addArgument(po::value<bool>(&saveAsVideo)->default_value(false)->implicit_value(true),         "video", "enable video mode (save as avi, x264)");
    argumentParser->addArgument(po::value<std::string>(&fileName)->default_value(ss.str()), "filename", "(only HeadLessRecorder) name of the file");
    argumentParser->addArgument(po::value<int>(&recordTimeInSeconds)->default_value(5), "recordTime", "(only HeadLessRecorder) seconds of recording, video or pictures of the simulation");
    argumentParser->addArgument(po::value<int>(&width)->default_value(1920), "width", "(only HeadLessRecorder) video or picture width");
    argumentParser->addArgument(po::value<int>(&height)->default_value(1080), "height", "(only HeadLessRecorder) video or picture height");
    argumentParser->addArgument(po::value<int>(&fps)->default_value(60), "fps", "(only HeadLessRecorder) define how many frame per second HeadlessRecorder will generate");
    argumentParser->addArgument(po::value<bool>(&recordUntilStopAnimate)->default_value(false)->implicit_value(true),         "recordUntilEndAnimate", "(only HeadLessRecorder) recording until the end of animation does not care how many seconds have been set");
    argumentParser->addArgument(po::value<std::string>(&recordTypeRaw)->default_value("wallclocktime"), "recordingmode", "(only HeadLessRecorder) define how the recording should be made; either \"simulationtime\" (records as if it was simulating in real time and skips frames accordingly), \"wallclocktime\" (records a frame for each time step) or an arbitrary interval time between each frame as a float.");
    return 0;
}

BaseGUI* HeadlessRecorder::CreateGUI(const char* /*name*/, sofa::simulation::Node::SPtr groot, const char* filename)
{
    msg_warning("HeadlessRecorder") << "This is an experimental feature. Works only on linux.\n\t" << "For any suggestion/help/bug please report to:\n\t" << "https://github.com/sofa-framework/sofa/pull/538";

    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        None
    };

    Display* m_display;
    int fbcount = 0;
    GLXFBConfig* fbc = NULL;
    GLXContext ctx;
    GLXPbuffer pbuf;

    /* open display */
    if ( ! (m_display = XOpenDisplay(0)) ){
        fprintf(stderr, "Failed to open display\n");
        exit(1);
    }

    /* get framebuffer configs, any is usable (might want to add proper attribs) */
    if ( !(fbc = glXChooseFBConfig(m_display, DefaultScreen(m_display), NULL, &fbcount) ) ){
        fprintf(stderr, "Failed to get FBConfig\n");
        exit(1);
    }

    /* get the required extensions */
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB");
    glXMakeContextCurrentARB = (glXMakeContextCurrentARBProc)glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent");
    if ( !(glXCreateContextAttribsARB && glXMakeContextCurrentARB) ){
        fprintf(stderr, "missing support for GLX_ARB_create_context\n");
        XFree(fbc);
        exit(1);
    }

    /* create a context using glXCreateContextAttribsARB */
    if ( !( ctx = glXCreateContextAttribsARB(m_display, fbc[0], 0, True, context_attribs)) ){
        fprintf(stderr, "Failed to create opengl context\n");
        XFree(fbc);
        exit(1);
    }

    /* create temporary pbuffer */
    int pbuffer_attribs[] = {
        GLX_PBUFFER_WIDTH, width,
        GLX_PBUFFER_HEIGHT, height,
        None
    };
    pbuf = glXCreatePbuffer(m_display, fbc[0], pbuffer_attribs);

    XFree(fbc);
    XSync(m_display, False);

    /* try to make it the current context */
    if ( !glXMakeContextCurrent(m_display, pbuf, pbuf, ctx) ){
        /* some drivers does not support context without default framebuffer, so fallback on
                    * using the default window.
                    */
        if ( !glXMakeContextCurrent(m_display, DefaultRootWindow(m_display), DefaultRootWindow(m_display), ctx) ){
            fprintf(stderr, "failed to make current\n");
            exit(1);
        }
    }

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        std:: cout << "GLEW Error: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    HeadlessRecorder* gui = new HeadlessRecorder();
    gui->setScene(groot, filename);
    gui->initializeGL();
    return gui;
}

int HeadlessRecorder::closeGUI()
{
    delete this;
    return 0;
}

// -----------------------------------------------------------------
// --- OpenGL stuff
// -----------------------------------------------------------------
void HeadlessRecorder::initializeGL(void)
{
    static GLfloat    specular[4];
    static GLfloat    ambientLight[4];
    static GLfloat    diffuseLight[4];
    static GLfloat    lightPosition[4];
    static GLfloat    lmodel_ambient[]    = {0.0f, 0.0f, 0.0f, 0.0f};
    static GLfloat    lmodel_twoside[]    = {GL_FALSE};
    static GLfloat    lmodel_local[]        = {GL_FALSE};
    static bool       initialized            = false;

    if (!initialized)
    {
        lightPosition[0] = -0.7f;
        lightPosition[1] = 0.3f;
        lightPosition[2] = 0.0f;
        lightPosition[3] = 1.0f;

        ambientLight[0] = 0.5f;
        ambientLight[1] = 0.5f;
        ambientLight[2] = 0.5f;
        ambientLight[3] = 1.0f;

        diffuseLight[0] = 0.9f;
        diffuseLight[1] = 0.9f;
        diffuseLight[2] = 0.9f;
        diffuseLight[3] = 1.0f;

        specular[0] = 1.0f;
        specular[1] = 1.0f;
        specular[2] = 1.0f;
        specular[3] = 1.0f;

        // Set light model
        glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, lmodel_local);
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

        // Setup 'light 0'
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
        glEnable(GL_LIGHT0);

        // Define background color
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // frame buffer
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // color render buffer
        glGenRenderbuffers(1, &rbo_color);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_color);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, width, height);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_color);

        /* Depth renderbuffer. */
        glGenRenderbuffers(1, &rbo_depth);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);
        glReadBuffer(GL_COLOR_ATTACHMENT0);

        glEnable(GL_DEPTH_TEST);

        initialized = true;
    }

    // switch to preset view
    resetView();
}

bool HeadlessRecorder::canRecord()
{
    if(recordUntilStopAnimate)
    {
        return currentSimulation() && currentSimulation()->getContext()->getAnimate();
    }
    return static_cast<float>(m_nFrames)/static_cast<float>(fps) <= recordTimeInSeconds;
}

int HeadlessRecorder::mainLoop()
{
    // Boost program_option doesn't take the order or the options inter-dependencies into account,
    // so we parse this option after we are certain everythin was parsed.
    parseRecordingModeOption();

    if(currentCamera)
        currentCamera->setViewport(width, height);
    calcProjection();

    if (!saveAsVideo && !saveAsScreenShot)
    {
        msg_error("HeadlessRecorder") <<  "Please, use at least one option: picture or video mode.";
        return 0;
    }
    if ((recordType == RecordMode::simulationtime || recordType == RecordMode::timeinterval) && groot->getDt() > skipTime)
    {
        msg_error("HeadlessRecorder") << "Scene delta time (" << groot->getDt() << "s) is too big to provide images at the supplied fps; it should be at least <" << skipTime ;
        return 0;
    }


    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    while(canRecord())
    {
        if (keepFrame())
	{
            redraw();
            m_nFrames++;
            if(m_nFrames % fps == 0)
            {
                end = std::chrono::system_clock::now();
                int elapsed_milliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
                msg_info("HeadlessRecorder") << "Encoding : " << m_nFrames/fps << " seconds. Encoding time : " << elapsed_milliSeconds << " ms";
                start = std::chrono::system_clock::now();
            }
            record();
	}

        if (currentSimulation() && currentSimulation()->getContext()->getAnimate())
        {
            step();
        }
        else
        {
            sleep(0.01);
	}
    }
    msg_info("HeadlessRecorder") << "Recording time: " << recordTimeInSeconds << " seconds at: " << fps << " fps.";
    return 0;
}

bool HeadlessRecorder::keepFrame()
{
    switch(recordType)
    {
        case RecordMode::wallclocktime :
            return true;
        case RecordMode::simulationtime :
        case RecordMode::timeinterval :
            return groot->getTime() >= m_nFrames * skipTime;
    }
    return false;
}

void HeadlessRecorder::redraw()
{
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    paintGL();
    glFlush();
}

void HeadlessRecorder::displayOBJs()
{
    vparams->sceneBBox() = groot->f_bbox.getValue();
    if (!initTexturesDone)
    {
        simulation::getSimulation()->initTextures(groot.get());
        initTexturesDone = true;
    } else
    {
        simulation::getSimulation()->draw(vparams,groot.get());
    }
}

void HeadlessRecorder::drawScene(void)
{
    if (!groot) return;
    if(!currentCamera)
    {
        msg_error("HeadlessRecorder") << "ERROR: no camera defined";
        return;
    }

    calcProjection();
    glLoadIdentity();

    GLdouble mat[16];
    currentCamera->getOpenGLModelViewMatrix(mat);
    glMultMatrixd(mat);
    displayOBJs();
}

void HeadlessRecorder::calcProjection()
{
    double xNear, yNear;
    double xFactor = 1.0, yFactor = 1.0;
    double offset;
    double xForeground, yForeground, zForeground, xBackground, yBackground, zBackground;
    Vector3 center;

    /// Camera part
    if (!currentCamera)
        return;

    if (groot && (!groot->f_bbox.getValue().isValid()))
    {
        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());
    }
    currentCamera->computeZ();

    vparams->zNear() = currentCamera->getZNear();
    vparams->zFar() = currentCamera->getZFar();

    xNear = 0.35 * vparams->zNear();
    yNear = 0.35 * vparams->zNear();
    offset = 0.001 * vparams->zNear(); // for foreground and background planes

    if ((height != 0) && (width != 0))
    {
        if (height > width)
        {
            xFactor = 1.0;
            yFactor = (double) height / (double) width;
        }
        else
        {
            xFactor = (double) width / (double) height;
            yFactor = 1.0;
        }
    }
    vparams->viewport() = sofa::helper::make_array(0,0,width,height);

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    xFactor *= 0.01;
    yFactor *= 0.01;

    zForeground = -vparams->zNear() - offset;
    zBackground = -vparams->zFar() + offset;

    if (currentCamera->getCameraType() == core::visual::VisualParams::PERSPECTIVE_TYPE)
        gluPerspective(currentCamera->getFieldOfView(), (double) width / (double) height, vparams->zNear(), vparams->zFar());
    else
    {
        float ratio = (float)( vparams->zFar() / (vparams->zNear() * 20) );
        Vector3 tcenter = vparams->sceneTransform() * center;
        if (tcenter[2] < 0.0)
        {
            ratio = (float)( -300 * (tcenter.norm2()) / tcenter[2] );
        }
        glOrtho((-xNear * xFactor) * ratio, (xNear * xFactor) * ratio, (-yNear * yFactor) * ratio, (yNear * yFactor) * ratio,
                vparams->zNear(), vparams->zFar());
    }

    xForeground = -zForeground * xNear / vparams->zNear();
    yForeground = -zForeground * yNear / vparams->zNear();
    xBackground = -zBackground * xNear / vparams->zNear();
    yBackground = -zBackground * yNear / vparams->zNear();

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;

    glGetDoublev(GL_PROJECTION_MATRIX,lastProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);
}

void HeadlessRecorder::paintGL()
{
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClearDepth(1.0);
    drawScene();
}

void HeadlessRecorder::step()
{
#ifdef SOFA_SMP
    mg->step();
#else
    getSimulation()->animate(groot.get());
#endif
    getSimulation()->updateVisual(groot.get());
    redraw();
}

void HeadlessRecorder::resetView()
{
    bool fileRead = false;

    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + ".view";
        fileRead = currentCamera->importParametersFromFile(viewFileName);
    }

    //if there is no .view file , look at the center of the scene bounding box
    // and with a Up vector in the same axis as the gravity
    if (!fileRead)
    {
        newView();
    }
    redraw();
}

void HeadlessRecorder::newView()
{
    if (!currentCamera || !groot)
        return;

    currentCamera->setDefaultView(groot->getGravity());
}

void HeadlessRecorder::setScene(sofa::simulation::Node::SPtr scene, const char* filename, bool)
{
    std::ostringstream ofilename;

    sceneFileName = (filename==NULL)?"":filename;
    if (!sceneFileName.empty())
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr(begin,'.');
        if (!end) end = begin + sceneFileName.length();
        ofilename << std::string(begin, end);
        ofilename << "_";
    }
    else
        ofilename << "scene_";

    groot = scene;
    initTexturesDone = false;

    //Camera initialization
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = sofa::core::objectmodel::New<component::visualmodel::InteractiveCamera>();
            currentCamera->setName(core::objectmodel::Base::shortName(currentCamera.get()));
            groot->addObject(currentCamera);
            currentCamera->p_position.forceSet();
            currentCamera->p_orientation.forceSet();
            currentCamera->bwdInit();
            resetView();
        }

        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());

    }
    redraw();
}

sofa::simulation::Node* HeadlessRecorder::currentSimulation()
{
    return groot.get();
}

void HeadlessRecorder::setViewerResolution(int /*width*/, int /*height*/)
{
}

// -----------------------------------------------------------------
// --- FrameRecord
// -----------------------------------------------------------------
void HeadlessRecorder::record()
{

    if (saveAsScreenShot)
    {
        std::string pngFilename = fileName + std::to_string(m_nFrames) + ".png" ;
        screenshotPNG(pngFilename);
    } else if (saveAsVideo)
    {
        if (initVideoRecorder)
        {
            std::string videoFilename = fileName;
            videoFilename.append(".avi");
            videoEncoderStart(videoFilename.c_str(), AV_CODEC_ID_H264);
            initVideoRecorder = false;
        }
        videoGLToFrame();
        if (canRecord())
            videoFrameEncoder();
        else
            videoEncoderStop();
    }
}

// -----------------------------------------------------------------
// --- Screenshot
// -----------------------------------------------------------------
void HeadlessRecorder::screenshotPNG(std::string filename)
{
    std::string extension = sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    //test if we can export in lossless image
    bool imageSupport = helper::io::Image::FactoryImage::getInstance()->hasKey(extension);
    if (!imageSupport)
    {
        msg_error("Capture") << "Could not write " << extension << "image format (no support found)";
        return;
    }
    helper::io::Image* img = helper::io::Image::FactoryImage::getInstance()->createObject(extension, "");
    bool success = false;
    if (img)
    {
        img->init(width, height, 1, 1, sofa::helper::io::Image::UNORM8, sofa::helper::io::Image::RGBA);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, img->getPixels());

        success = img->save(filename, 0);

        if (success)
        {
            msg_info("Capture") << "Saved " << img->getWidth() << "x" << img->getHeight() << " screen image to " << filename;
        }
        delete img;
    }

    if(!success)
    {
        msg_error("Capture") << "Unknown error while saving screen image to " << filename;
    }
}

// -----------------------------------------------------------------
// --- Screencast
// -----------------------------------------------------------------

void HeadlessRecorder::videoEncoderStart(const char *filename, int codec_id)
{
    AVCodec *codec;
    int ret;
    msg_info("HeadlessRecorder") << "Start recording ... " << filename;
    avcodec_register_all();
    codec = avcodec_find_encoder((AVCodecID)codec_id);
    if (!codec) {
        msg_error("HeadlessRecorder") << "Codec not found";
        exit(1);
    }
    c = avcodec_alloc_context3(codec);
    if (!c) {
        msg_error("HeadlessRecorder") << "Could not allocate video codec context";
        exit(1);
    }

    m_avPacket = av_packet_alloc();
    if (!m_avPacket)
        exit(1);

    c->bit_rate = 8000000; // maybe I need to adjust it
    c->width = width;
    c->height = height;
    c->time_base = (AVRational){1, fps};
    c->framerate = (AVRational){fps, 1};
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (codec_id == AV_CODEC_ID_H264)
        av_opt_set(c->priv_data, "preset", "slow", 0);

    if (avcodec_open2(c, codec, NULL) < 0)
    {
        msg_error("HeadlessRecorder") << "Could not open codec";
        exit(1);
    }
    m_file = fopen(filename, "wb");
    if (!m_file) {
        msg_error("HeadlessRecorder") << "Could not open " << filename;
        exit(1);
    }

    m_frame = av_frame_alloc();
    if (!m_frame)
    {
        msg_error("HeadlessRecorder") << "Could not allocate video frame";
        exit(1);
    }
    m_frame->format = c->pix_fmt;
    m_frame->width  = c->width;
    m_frame->height = c->height;

    ret = av_frame_get_buffer(m_frame, 32);
    if (ret < 0)
    {
        msg_error("HeadlessRecorder") << "Could not allocate the video frame data";
        exit(1);
    }
}

void HeadlessRecorder::encode()
{
    int ret;
    m_frame->pts = m_nFrames;
    ret = avcodec_send_frame(c, m_frame);
    if (ret < 0) {
        msg_error("HeadlessRecorder") << "Error sending a frame for encoding";
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(c, m_avPacket);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            msg_error("HeadlessRecorder") << "Error during encoding";
            exit(1);
        }
        fwrite(m_avPacket->data, 1, m_avPacket->size, m_file);
        av_packet_unref(m_avPacket);
    }
}

void HeadlessRecorder::videoFrameEncoder()
{
    int ret;
    ret = av_frame_make_writable(m_frame);
    if (ret < 0)
    {
        msg_error("HeadlessRecorder") << "Frame is not writable";
        exit(1);
    }
    videoYUVToRGB();
    encode();
}

void HeadlessRecorder::videoEncoderStop()
{
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };
    encode();

    fwrite(endcode, 1, sizeof(endcode), m_file);
    fclose(m_file);

    avcodec_free_context(&c);
    av_frame_free(&m_frame);
    av_packet_free(&m_avPacket);
}

void HeadlessRecorder::videoGLToFrame()
{
    int i, j, k, cur_gl, cur_rgb, nvals;
    const int format_nchannels = 4;
    nvals = format_nchannels * width * height;
    m_rgb = (uint8_t*)realloc(m_rgb, nvals * sizeof(uint8_t));
    GLubyte *pixels = (GLubyte*)malloc(nvals * sizeof(GLubyte));
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            cur_gl  = format_nchannels * (width * (height - i - 1) + j);
            cur_rgb = format_nchannels * (width * i + j);
            for (k = 0; k < format_nchannels; k++)
                (m_rgb)[cur_rgb + k] = (pixels)[cur_gl + k];
        }
    }
    free(pixels);
}

void HeadlessRecorder::videoYUVToRGB() {
    const int in_linesize[1] = { 4 * c->width };
    sws_context = sws_getCachedContext(sws_context,
                                       c->width, c->height, AV_PIX_FMT_RGBA,
                                       c->width, c->height, AV_PIX_FMT_YUV420P,
                                       0, NULL, NULL, NULL);

    sws_scale(sws_context, (const uint8_t * const *)&m_rgb, in_linesize, 0, c->height, m_frame->data, m_frame->linesize);
}

} // namespace glut

} // namespace gui

} // namespace sofa

