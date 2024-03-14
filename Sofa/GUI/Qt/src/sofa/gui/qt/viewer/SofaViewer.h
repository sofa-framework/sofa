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
#pragma once
#include <sofa/gui/qt/config.h>

#include <sofa/gui/common/BaseViewer.h>
#include <sofa/gui/qt/PickHandlerCallBacks.h>
#include <sofa/gui/qt/SofaVideoRecorderManager.h>
#include <sofa/gui/qt/viewer/EngineBackend.h>

#include <QString>
#include <QWidget>

#include <QEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QTabWidget>
#include <QTimer>

namespace sofa::gui::qt::viewer
{

enum
{
    BTLEFT_MODE = 101, BTRIGHT_MODE = 102, BTMIDDLE_MODE = 103,
};

class SOFA_GUI_QT_API SofaViewer : public sofa::gui::common::BaseViewer
{

public:
    SofaViewer();
    ~SofaViewer() override;

    /// Optional QTabWidget GUI for a concreate viewer.
    virtual void removeViewerTab(QTabWidget *) {}
    /// Optional QTabWidget GUI for a concreate viewer.
    virtual void configureViewerTab(QTabWidget *) {}

    virtual QWidget* getQWidget()=0;
    virtual QString helpString() const =0;

    //*************************************************************
    // QT
    //*************************************************************
    //SLOTS
    void captureEvent() override;

    // ---------------------- Here are the Keyboard controls   ----------------------
    virtual void keyPressEvent(QKeyEvent * e);
    virtual void keyReleaseEvent(QKeyEvent * e);
    bool isControlPressed() const;
    // ---------------------- Here are the Mouse controls   ----------------------
    virtual void wheelEvent(QWheelEvent *e);
    virtual void mouseMoveEvent ( QMouseEvent *e );
    virtual void mousePressEvent ( QMouseEvent * e);
    virtual void mouseReleaseEvent ( QMouseEvent * e);
    virtual bool mouseEvent(QMouseEvent *e);

    // Overriden from BaseViewer
    virtual void configure(sofa::component::setting::ViewerSetting* viewerConf) override;
    const std::string screenshotName() override;
    void setPrefix(const std::string& prefix, bool prependDirectory = true) override;
    virtual void screenshot(const std::string& filename, int compression_level =-1) override;
    virtual void setBackgroundImage(std::string imageFileName = std::string("textures/SOFA_logo.bmp")) override;

protected:
    std::unique_ptr<EngineBackend> m_backend;

    void redraw() override;

    QTimer captureTimer;

    bool m_isControlPressed;

    ColourPickingRenderCallBack colourPickingRenderCallBack;

signals:
    virtual void redrawn() = 0;
    virtual void resizeW(int) = 0;
    virtual void resizeH(int) = 0;
};

} // namespace sofa::gui::qt::viewer
