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
#include <sofa/gui/common/config.h>

#include <sofa/gui/common/ColourPickingVisitor.h>

#include <sofa/helper/Factory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/visual/BaseCamera.h>

#include <string>

namespace sofa::component::setting
{
    class ViewerSetting;

} // namespace sofa::component::setting

namespace sofa::gui::common
{

class PickHandler;

enum
{
    BTLEFT_MODE = 101, BTRIGHT_MODE = 102, BTMIDDLE_MODE = 103,
};


class SOFA_GUI_COMMON_API BaseViewer
{

public:
    BaseViewer();
    virtual ~BaseViewer();

    virtual void drawColourPicking (ColourPickingVisitor::ColourCode /*code*/) {}

    virtual sofa::simulation::Node* getScene();
    virtual const std::string& getSceneFileName();
    virtual void setSceneFileName(const std::string &f);
    virtual void setScene(sofa::simulation::Node::SPtr scene, const char* filename = nullptr, bool /*keepParams*/= false);
    virtual void setCameraMode(core::visual::VisualParams::CameraType);

    /// true when the viewer keep the hand on the render
    /// false when it's not in activity
    virtual bool ready();

    /// ask the viewer to resume its activity
    virtual void wait();

    /// Load the viewer. It's the initialisation
    virtual bool load(void);

    /// unload the viewer without delete
    virtual bool unload(void);

    /// Recompute viewer's home position so it encompass the whole scene and apply it
    virtual void viewAll(void) = 0;

    //Allow to configure your viewer using the Sofa Component, ViewerSetting
    virtual void configure(sofa::component::setting::ViewerSetting* viewerConf);

    //Functions needed to take a screenshot
    virtual const std::string screenshotName();
    virtual void setPrefix(const std::string& prefix, bool prependDirectory = true);
    virtual void screenshot(const std::string& filename, int compression_level =-1);

    virtual void getView(sofa::type::Vec3& pos, sofa::type::Quat<SReal>& ori) const;
    virtual void setView(const sofa::type::Vec3& pos, const sofa::type::Quat<SReal> &ori);
    virtual void moveView(const sofa::type::Vec3& pos, const sofa::type::Quat<SReal> &ori);
    virtual void newView();
    virtual void resetView();

    virtual void setBackgroundColour(float r, float g, float b);
    virtual void setBackgroundImage(std::string imageFileName = std::string("textures/SOFA_logo.bmp"));
    std::string getBackgroundImage();

    virtual void saveView()=0;
    virtual void setSizeW(int)=0;
    virtual void setSizeH(int)=0;
    virtual int getWidth()=0;
    virtual int getHeight()=0;
    virtual void captureEvent() {}
    virtual void fitObjectBBox(sofa::core::objectmodel::BaseObject* );
    virtual void fitNodeBBox(sofa::core::objectmodel::BaseNode*);

    virtual void setFullScreen(bool /*enable*/) {}


    /// RayCasting PickHandler
    virtual void moveRayPickInteractor(int, int) {}

    PickHandler* getPickHandler();

    /// the rendering pass is done here (have to be called in a loop)
    virtual void drawScene(void) = 0;

    void drawSelection(sofa::core::visual::VisualParams* vparams);

    void setCurrentSelection(const std::set<core::objectmodel::Base::SPtr> &selection);
    const std::set<sofa::core::objectmodel::Base::SPtr>& getCurrentSelection() const;

public:
    bool m_enableSelectionDraw {true};
    bool m_showSelectedNodeBoundingBox {true};
    bool m_showSelectedObjectBoundingBox {true};
    bool m_showSelectedObjectPositions {false};
    bool m_showSelectedObjectSurfaces {false};
    bool m_showSelectedObjectVolumes {false};
    bool m_showSelectedObjectIndices {false};
    float m_visualScaling {0.2};

protected:
    void drawIndices(const sofa::type::BoundingBox& bbox, const std::vector<sofa::type::Vec3>& positions);

    /// internally called while the actual viewer needs a redraw (ie the camera changed)
    virtual void redraw() = 0;

    /// the sofa root note of the current scene
    sofa::simulation::Node::SPtr groot;

    sofa::component::visual::BaseCamera::SPtr currentCamera;

    std::string sceneFileName;

    bool _video;
    bool m_isVideoButtonPressed;
    bool m_bShowAxis;

    bool _fullScreen;
    int _background;
    bool initTexturesDone;

    sofa::type::Vec3 backgroundColour;
    std::string backgroundImageFile;

    sofa::type::Vec3 ambientColour;

    std::unique_ptr<PickHandler> pick;

    //instruments handling
    int _navigationMode;
    bool _mouseInteractorMoving;
    int _mouseInteractorSavedPosX;
    int _mouseInteractorSavedPosY;

    std::string _screenshotDirectory;

    std::set<sofa::core::objectmodel::Base::SPtr> currentSelection;
};

} // namespace sofa::gui::common
