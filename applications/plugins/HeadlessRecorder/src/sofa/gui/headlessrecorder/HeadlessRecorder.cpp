/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/Utils.h>
using sofa::helper::Utils;
#include <sofa/helper/system/SetDirectory.h>
using sofa::helper::system::SetDirectory;

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/gl/Texture.h>

#include <sofa/gui/common/ArgumentParser.h>
#include <sofa/component/visual/InteractiveCamera.h>
#include <thread>
#include <chrono>

namespace sofa::gui::hrecorder
{

GLsizei HeadlessRecorder::s_width = 1920;
GLsizei HeadlessRecorder::s_height = 1080;
int HeadlessRecorder::recordTimeInSeconds = 5;
unsigned int HeadlessRecorder::fps = 60;
std::string HeadlessRecorder::fileName = "tmp";
bool HeadlessRecorder::saveAsVideo = false;
bool HeadlessRecorder::saveAsScreenShot = false;
bool HeadlessRecorder::recordUntilStopAnimate = false;

std::string HeadlessRecorder::recordTypeRaw = "wallclocktime";
RecordMode HeadlessRecorder::recordType = RecordMode::wallclocktime;
float HeadlessRecorder::skipTime = 0;

using namespace sofa::type;
using namespace sofa::defaulttype;
using sofa::simulation::getSimulation;

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = nullptr;
static glXMakeContextCurrentARBProc glXMakeContextCurrentARB = nullptr;

void static_handler(int /*signum*/)
{
    HeadlessRecorder::recordUntilStopAnimate = false;
    HeadlessRecorder::recordTimeInSeconds = 0;
}

// Class
HeadlessRecorder::HeadlessRecorder()
    : groot(nullptr)
    , m_nFrames(0)
    , initTexturesDone(false)
    , requestVideoRecorderInit(true)
    , m_backgroundColor{0,0,0,0}
{
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

int HeadlessRecorder::RegisterGUIParameters(common::ArgumentParser* argumentParser)
{
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    //ss << std::put_time(std::localtime(&in_time_t), "%F-%X");
    ss << std::put_time(std::localtime(&in_time_t), "%F-%H %M %S");

    argumentParser->addArgument(cxxopts::value<bool>(saveAsScreenShot)->default_value("false")->implicit_value("true"),
                                "picture", "enable picture mode (save as png)");
    argumentParser->addArgument(cxxopts::value<bool>(saveAsVideo)->default_value("false")->implicit_value("true"),
                                "video", "enable video mode (save as avi, x264)");
    argumentParser->addArgument(cxxopts::value<std::string>(fileName)->default_value(ss.str()),
                                "filename", "(only HeadLessRecorder) name of the file");
    argumentParser->addArgument(cxxopts::value<int>(recordTimeInSeconds)->default_value("5"),
                                "recordTime", "(only HeadLessRecorder) seconds of recording, video or pictures of the simulation");
    argumentParser->addArgument(cxxopts::value<GLsizei>(s_width)->default_value("1920"),
                                "width", "(only HeadLessRecorder) video or picture width");
    argumentParser->addArgument(cxxopts::value<GLsizei>(s_height)->default_value("1080"),
                                "height", "(only HeadLessRecorder) video or picture height");
    argumentParser->addArgument(cxxopts::value<unsigned int>(fps)->default_value("60"),
                                "fps", "(only HeadLessRecorder) define how many frame per second HeadlessRecorder will generate");
    argumentParser->addArgument(cxxopts::value<bool>(recordUntilStopAnimate)->default_value("false")->implicit_value("true"),
                                "recordUntilEndAnimate", "(only HeadLessRecorder) recording until the end of animation does not care how many seconds have been set");
    argumentParser->addArgument(cxxopts::value<std::string>(recordTypeRaw)->default_value("wallclocktime"),
                                "recordingmode", "(only HeadLessRecorder) define how the recording should be made; either \"simulationtime\" (records as if it was simulating in real time and skips frames accordingly), \"wallclocktime\" (records a frame for each time step) or an arbitrary interval time between each frame as a float.");

    return 0;
}

common::BaseGUI* HeadlessRecorder::CreateGUI(const char* name, sofa::simulation::Node::SPtr groot, const char* filename)
{
    SOFA_UNUSED(name);
    SOFA_UNUSED(groot);
    SOFA_UNUSED(filename);
    msg_warning("HeadlessRecorder") << "This is an experimental feature. \n\t" << "For any suggestion/help/bug please report to:\n\t" << "https://github.com/sofa-framework/sofa/pull/538";

    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        None
    };

    /* open display */
    Display* m_display = XOpenDisplay(nullptr);
    if (!m_display){
        msg_error("HeadlessRecorder") << "Failed to open display";
        exit(EXIT_FAILURE);
    }

    /* get framebuffer configs, any is usable (might want to add proper attribs) */
    int fbcount = 0;
    GLXFBConfig* fbc = glXChooseFBConfig(m_display, DefaultScreen(m_display), nullptr, &fbcount);
    if (!fbc){
        msg_error("HeadlessRecorder") << "Failed to get FBConfig";
        exit(EXIT_FAILURE);
    }

    /* get the required extensions */
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB");
    glXMakeContextCurrentARB = (glXMakeContextCurrentARBProc)glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent");
    if ( !(glXCreateContextAttribsARB && glXMakeContextCurrentARB) ){
        msg_error("HeadlessRecorder") << "Missing support for GLX_ARB_create_context";
        XFree(fbc);
        exit(EXIT_FAILURE);
    }

    /* create a context using glXCreateContextAttribsARB */
    GLXContext ctx = glXCreateContextAttribsARB(m_display, fbc[0], nullptr, True, context_attribs);
    if (!ctx){
        msg_error("HeadlessRecorder") << "Failed to create opengl context";
        XFree(fbc);
        exit(EXIT_FAILURE);
    }

    /* create temporary pbuffer */
    int pbuffer_attribs[] = {
        GLX_PBUFFER_WIDTH, s_width,
        GLX_PBUFFER_HEIGHT, s_height,
        None
    };
    GLXPbuffer pbuf = glXCreatePbuffer(m_display, fbc[0], pbuffer_attribs);

    XFree(fbc);
    XSync(m_display, False);

    /* try to make it the current context */
    if ( !glXMakeContextCurrent(m_display, pbuf, pbuf, ctx) ){
        /* some drivers does not support context without default framebuffer, so fallback on
                    * using the default window.
                    */
        if ( !glXMakeContextCurrent(m_display, DefaultRootWindow(m_display), DefaultRootWindow(m_display), ctx) ){
            msg_error("HeadlessRecorder") << "Failed to make current";
            exit(EXIT_FAILURE);
        }
    }

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        msg_error("HeadlessRecorder") << "GLEW Error: " << glewGetErrorString(err);
        exit(EXIT_FAILURE);
    }

    HeadlessRecorder* gui = new HeadlessRecorder();
    //gui->setScene(groot, filename);
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
    const GLfloat specref[] {1.0f, 1.0f, 1.0f, 1.0f};
    const GLfloat specular[] {1.0f, 1.0f, 1.0f, 1.0f};
    const GLfloat ambientLight[] {0.5f, 0.5f, 0.5f, 1.0f};
    const GLfloat diffuseLight[] {0.9f, 0.9f, 0.9f, 1.0f};
    const GLfloat lightPosition[] {-0.7f, 0.3f, 0.0f, 1.0f};
    const GLfloat lmodel_ambient[] {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat lmodel_twoside[] {GL_FALSE};
    const GLfloat lmodel_local[] {GL_FALSE};
    static bool initialized = false;

    if (!initialized)
    {
        glDepthFunc(GL_LEQUAL);
        glClearDepth(1.0);
        glEnable(GL_NORMALIZE);

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);





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

        // Enable color tracking
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // All materials hereafter have full specular reflectivity with a high shine
        glMaterialfv(GL_FRONT, GL_SPECULAR, specref);
        glMateriali(GL_FRONT, GL_SHININESS, 128);

        glShadeModel(GL_SMOOTH);






        // frame buffer
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // color render buffer
        glGenRenderbuffers(1, &rbo_color);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_color);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, s_width, s_height);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_color);

        /* Depth renderbuffer. */
        glGenRenderbuffers(1, &rbo_depth);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, s_width, s_height);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);
        glReadBuffer(GL_COLOR_ATTACHMENT0);

        glEnable(GL_DEPTH_TEST);

        initialized = true;
    }

    // switch to preset view
    //resetView();
}

bool HeadlessRecorder::canRecord() const
{
    if(recordUntilStopAnimate)
    {
        return groot != nullptr && groot->getContext()->getAnimate();
    }
    return static_cast<float>(m_nFrames)/static_cast<float>(fps) <= recordTimeInSeconds;
}

int HeadlessRecorder::mainLoop()
{
    // Boost program_option doesn't take the order or the options inter-dependencies into account,
    // so we parse this option after we are certain everything else was parsed.
    parseRecordingModeOption();

    if(currentCamera)
        currentCamera->setViewport(s_width, s_height);
    calcProjection();

    if (!saveAsVideo && !saveAsScreenShot)
    {
        msg_error("HeadlessRecorder") << "Please, use at least one option: picture or video mode.";
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
                const auto elapsed_milliSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
                msg_info("HeadlessRecorder") << "Encoding : " << m_nFrames/fps << " seconds. Encoding time : " << elapsed_milliSeconds << " ms";
                start = std::chrono::system_clock::now();
            }
            record();
        }
        // getAnimate is false unless animate console line flag is provided.
        // If not provided, code is stuck in an infinite loop...
        if (currentSimulation()) // && currentSimulation()->getContext()->getAnimate())
        {
            step();
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);
            std::cout <<"vport " << viewport[0] << " "<< viewport[1] << " "<< viewport[2] << " "<< viewport[3] << std::endl;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    msg_info("HeadlessRecorder") << "Recording time: " << recordTimeInSeconds << " seconds at: " << fps << " fps.";
    return 0;
}

bool HeadlessRecorder::keepFrame() const
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
        sofa::simulation::node::initTextures(groot.get());
        initTexturesDone = true;
    } else
    {
        sofa::simulation::node::draw(vparams,groot.get());
    }
}

void HeadlessRecorder::drawScene(void)
{
    if (!groot) return;
    if(!currentCamera)
    {
        msg_error("HeadlessRecorder") << "No camera defined";
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
    double xFactor = 1.0, yFactor = 1.0;
    //double offset;
    //double xForeground, yForeground, zForeground, xBackground, yBackground, zBackground;
    Vec3 center;

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

    double xNear = 0.35 * vparams->zNear();
    double yNear = 0.35 * vparams->zNear();
    //offset = 0.001 * vparams->zNear(); // for foreground and background planes

    if ((s_height != 0) && (s_width != 0))
    {
        if (s_height > s_width)
        {
            xFactor = 1.0;
            yFactor = (double) s_height / (double) s_width;
        }
        else
        {
            xFactor = (double) s_width / (double) s_height;
            yFactor = 1.0;
        }
    }
    vparams->viewport() = sofa::type::make_array(0, 0, s_width, s_height);

    glViewport(0, 0, s_width, s_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    xFactor *= 0.01;
    yFactor *= 0.01;

    //zForeground = -vparams->zNear() - offset;
    //zBackground = -vparams->zFar() + offset;

    if (currentCamera->getCameraType() == core::visual::VisualParams::PERSPECTIVE_TYPE)
        gluPerspective(currentCamera->getFieldOfView(), (double) s_width / (double) s_height, vparams->zNear(), vparams->zFar());
    else
    {
        float ratio = static_cast<float>( vparams->zFar() / (vparams->zNear() * 20) );
        Mat4x4d projMat;
        vparams->getProjectionMatrix(projMat.ptr());
        Mat4x4d modelMat;
        vparams->getModelViewMatrix(modelMat.ptr());
        Vec3 tcenter = (projMat * modelMat).transform(center);
        if (tcenter[2] < 0.0)
        {
            ratio = static_cast<float>( -300 * (tcenter.norm2()) / tcenter[2] );
        }
        glOrtho((-xNear * xFactor) * ratio, (xNear * xFactor) * ratio, (-yNear * yFactor) * ratio, (yNear * yFactor) * ratio,
                vparams->zNear(), vparams->zFar());
    }

    /*xForeground = -zForeground * xNear / vparams->zNear();
    yForeground = -zForeground * yNear / vparams->zNear();
    xBackground = -zBackground * xNear / vparams->zNear();
    yBackground = -zBackground * yNear / vparams->zNear();

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;
    */
    glGetDoublev(GL_PROJECTION_MATRIX,lastProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);
}

void HeadlessRecorder::paintGL()
{
    glClearColor(m_backgroundColor.r(), m_backgroundColor.g(), m_backgroundColor.b(), m_backgroundColor.a());
    glClearDepth(1.0);
    drawScene();
}

void HeadlessRecorder::step()
{
    sofa::helper::AdvancedTimer::begin("Animate");
    sofa::simulation::node::animate(groot.get());
    sofa::helper::AdvancedTimer::end("Animate");
    sofa::simulation::node::updateVisual(groot.get());
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

void HeadlessRecorder::setScene(sofa::simulation::Node::SPtr scene, const char* filename, bool temporaryFile)
{
    SOFA_UNUSED(temporaryFile);
    std::ostringstream ofilename;

    sceneFileName = (filename == nullptr) ? "" : filename;
    if (!sceneFileName.empty())
    {
        ofilename << sceneFileName.substr(0, sceneFileName.find('.')) << "_";
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
            currentCamera = sofa::core::objectmodel::New<component::visual::InteractiveCamera>();
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

void HeadlessRecorder::setViewerResolution(int width, int height)
{
    SOFA_UNUSED(width);
    SOFA_UNUSED(height);
}

// -----------------------------------------------------------------
// --- FrameRecord
// -----------------------------------------------------------------
void HeadlessRecorder::record()
{
    if (saveAsScreenShot)
    {
        std::stringstream ss;
        ss << std::setw(8) << std::setfill('0') << m_nFrames;

        std::string pngFilename = fileName + ss.str() + ".png" ;
        m_screencapture.saveScreen(pngFilename, 0);
    }
    else if (saveAsVideo)
    {
        if (requestVideoRecorderInit)
            initVideoRecorder();
        if (canRecord()) {
            //m_videorecorder->encodeFrame();
            m_videorecorder.addFrame();
        } else {
            //m_videorecorder->stop();
            m_videorecorder.finishVideo();
        }
    }
}

// See also GLBackend::initRecorder
void HeadlessRecorder::initVideoRecorder()
{
    std::string ffmpeg_exec_path = "";
    const std::string ffmpegIniFilePath = Utils::getSofaPathTo("etc/SofaHeadlessRecorder.ini");
    std::map<std::string, std::string> iniFileValues = Utils::readBasicIniFile(ffmpegIniFilePath);
    if (iniFileValues.find("FFMPEG_EXEC_PATH") != iniFileValues.end())
    {
        // get absolute path of FFMPEG executable
        ffmpeg_exec_path = SetDirectory::GetRelativeFromProcess( iniFileValues["FFMPEG_EXEC_PATH"].c_str() );
    }

    std::string videoFilename = fileName;
    int bitrate = 100000000;
    videoFilename.append(".avi");
    //videoFilename.append(".mp4");
    //m_videorecorder = std::unique_ptr<VideoRecorderFFmpeg>(new VideoRecorderFFmpeg(fps, width, height, videoFilename.c_str(), AV_CODEC_ID_H264));
    //std::string codec = "yuv420p";
    std::string codec = "yuv444p";
    m_videorecorder.init(ffmpeg_exec_path, videoFilename, s_width, s_height, fps, bitrate, codec);
    //m_videorecorder->start();
    requestVideoRecorderInit = false;
}

void HeadlessRecorder::setBackgroundColor(const type::RGBAColor &color)
{
    m_backgroundColor = color;
    glClearColor(m_backgroundColor.r(), m_backgroundColor.g(), m_backgroundColor.b(), m_backgroundColor.a());
}

} // namespace sofa::gui::hrecorder
