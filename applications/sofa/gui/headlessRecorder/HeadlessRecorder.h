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
#ifndef SOFA_GUI_GLUT_HEADLESSRECORDER_H
#define SOFA_GUI_GLUT_HEADLESSRECORDER_H

#include <sofa/gui/BaseGUI.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <SofaBaseVisual/InteractiveCamera.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/core/ObjectFactory.h>

// LIBAV
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

// OPENGL
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <EGL/egl.h>

// PNG
#include <png.h>

namespace sofa
{

namespace gui
{

namespace hRecorder
{

class HeadlessRecorder : public sofa::gui::BaseGUI
{

public:
    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DrawToolGL   DrawToolGL;

    HeadlessRecorder(const std::vector<std::string>& options = std::vector<std::string>());
    ~HeadlessRecorder();

    int mainLoop();

    void step();
    void redraw();
    void resetView();
    void saveView();
    void initializeGL();
    void paintGL();
    void setScene(sofa::simulation::Node::SPtr scene, const char* filename=NULL, bool temporaryFile=false);
    void newView();

    // Virtual from BaseGUI
    virtual sofa::simulation::Node* currentSimulation() override;
    virtual int closeGUI() override;
    virtual void setViewerResolution(int width, int height) override;

    // Needed for the registration
    static int InitGUI(const char* name, const std::vector<std::string>& options);
    static BaseGUI* CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);


private:

    void parseOptions(const std::vector<std::string>& options);

    void record();
    void screenshotPNG(std::string fileName);
    void videoYUVToRGB();
    void videoEncoderStart(const char *filename, int codec_id);
    void videoEncoderStop(void);
    void videoFrameEncoder();
    void videoGLToFrame();

    void displayOBJs();
    void drawScene();
    void calcProjection();

    void stopRecording() { m_isRecording = false;}
    void startRecording() { m_isRecording = true;}

    VisualParams* vparams;
    DrawToolGL   drawTool;

    sofa::simulation::Node::SPtr groot;
    std::string sceneFileName;
    sofa::component::visualmodel::BaseCamera::SPtr currentCamera;

    int m_width, m_height;
    int m_nFrames;
    int m_fps;
    int m_recordTimeInSeconds;
    FILE* m_file;

    AVCodecContext *c = NULL;
    AVFrame *m_frame;
    AVPacket m_avPacket;
    struct SwsContext *sws_context = NULL;
    uint8_t *m_rgb = NULL;

    GLuint fbo;
    GLuint rbo_color, rbo_depth;
    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];
    bool initTexturesDone;
    bool saveAsScreenShot, saveAsVideo, initVideoRecorder;
    bool m_isRecording;
    std::string m_fileNameVideo, m_fileNamePNG;

};

} // namespace glut

} // namespace gui

} // namespace sofa

#endif
