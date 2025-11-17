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
#include <sofa/gui/common/BaseViewer.h>
#include <sofa/gui/common/PickHandler.h>
#include <sofa/gui/common/BaseGUI.h>

#include <sofa/helper/Factory.inl>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/component/setting/ViewerSetting.h>
#include <sofa/component/visual/VisualStyle.h>
#include <sofa/component/visual/InteractiveCamera.h>

#include <sofa/core/ComponentNameHelper.h>
#include <sofa/helper/system/FileSystem.h>


namespace sofa::gui::common
{

BaseViewer::BaseViewer()
    : groot(nullptr)
    , currentCamera(nullptr)
    , _video(false)
    , m_isVideoButtonPressed(false)
    , m_bShowAxis(false)
    , backgroundColour(type::Vec3())
    , backgroundImageFile("textures/SOFA_logo.bmp")
    , ambientColour(type::Vec3())
    , pick(std::make_unique<PickHandler>())
    , _screenshotDirectory(".")
{}

BaseViewer::~BaseViewer()
{
    //Save parameter file to set selection behavior
    const std::string baseViewerFilename = BaseGUI::getConfigDirectoryPath() + "/BaseViewer.ini";
    std::ofstream out(baseViewerFilename.c_str(),std::ios::trunc);
    out<<std::boolalpha;
    out<<"EnableSelectionDraw="<<m_enableSelectionDraw<<std::endl;
    out<<"ShowSelectedNodeBoundingBox="<<m_showSelectedNodeBoundingBox<<std::endl;
    out<<"ShowSelectedObjectBoundingBox="<<m_showSelectedObjectBoundingBox<<std::endl;
    out<<"ShowSelectedObjectPositions="<<m_showSelectedObjectPositions<<std::endl;
    out<<"ShowSelectedObjectSurfaces="<<m_showSelectedObjectSurfaces<<std::endl;
    out<<"ShowSelectedObjectVolumes="<<m_showSelectedObjectVolumes<<std::endl;
    out<<"ShowSelectedObjectIndices="<<m_showSelectedObjectIndices<<std::endl;
    out<<"VisualScaling="<<m_visualScaling<<std::endl;
    out.close();
}

sofa::simulation::Node* BaseViewer::getScene()
{
    return groot.get();
}
const std::string& BaseViewer::getSceneFileName()
{
    return sceneFileName;
}
void BaseViewer::setSceneFileName(const std::string &f)
{
    sceneFileName = f;
}

void BaseViewer::setScene(sofa::simulation::Node::SPtr scene, const char* filename /* = nullptr */, bool /* = false */)
{
    std::string prefix = "";
    if (filename)
        prefix = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename) + "_";
    setPrefix(prefix);

    sceneFileName = filename ? filename : std::string("default.scn");
    groot = scene;
    initTexturesDone = false;
}

void BaseViewer::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    currentCamera->setCameraType(mode);
}

bool BaseViewer::ready()
{
    return true;
}

void BaseViewer::wait()
{
}

void BaseViewer::configure(sofa::component::setting::ViewerSetting* viewerConf)
{
    using namespace core::visual;
    if (viewerConf->d_cameraMode.getValue().getSelectedId() == VisualParams::ORTHOGRAPHIC_TYPE)
        setCameraMode(VisualParams::ORTHOGRAPHIC_TYPE);
    else
        setCameraMode(VisualParams::PERSPECTIVE_TYPE);
}

//Functions needed to take a screenshot
const std::string BaseViewer::screenshotName()
{
    return "";
}

void BaseViewer::setPrefix(const std::string& prefix, bool prependDirectory)
{
    SOFA_UNUSED(prefix);
    SOFA_UNUSED(prependDirectory);
}

void BaseViewer::screenshot(const std::string& filename, int compression_level)
{
    SOFA_UNUSED(filename);
    SOFA_UNUSED(compression_level);
}

void BaseViewer::getView(type::Vec3& pos, type::Quat<SReal>& ori) const
{
    if (!currentCamera)
        return;

    const type::Vec3d& camPosition = currentCamera->getPosition();
    const type::Quat<SReal>& camOrientation = currentCamera->getOrientation();

    pos[0] = camPosition[0];
    pos[1] = camPosition[1];
    pos[2] = camPosition[2];

    ori[0] = camOrientation[0];
    ori[1] = camOrientation[1];
    ori[2] = camOrientation[2];
    ori[3] = camOrientation[3];
}

void BaseViewer::setView(const type::Vec3& pos, const type::Quat<SReal> &ori)
{
    type::Vec3d position;
    type::Quat<SReal> orientation;
    for (unsigned int i=0 ; i<3 ; i++)
    {
        position[i] = pos[i];
        orientation[i] = ori[i];
    }
    orientation[3] = ori[3];

    if (currentCamera)
        currentCamera->setView(position, orientation);

    redraw();
}

void BaseViewer::moveView(const type::Vec3& pos, const type::Quat<SReal> &ori)
{
    if (!currentCamera)
        return;

    currentCamera->moveCamera(pos, ori);

    redraw();
}

void BaseViewer::newView()
{
    if (!currentCamera || !groot)
        return;

    currentCamera->setDefaultView(groot->getGravity());
}

void BaseViewer::resetView()
{
    redraw();
}

void BaseViewer::setBackgroundColour(float r, float g, float b)
{
    _background = 3;
    backgroundColour[0] = r;
    backgroundColour[1] = g;
    backgroundColour[2] = b;
}

void BaseViewer::setBackgroundImage(std::string imageFileName)
{
    SOFA_UNUSED(imageFileName);
}

std::string BaseViewer::getBackgroundImage()
{
    return backgroundImageFile;
}

PickHandler* BaseViewer::getPickHandler()
{
    return pick.get();
}

bool BaseViewer::load()
{
    //Load parameter file to set selection behavior
    const std::string baseViewerFilename = BaseGUI::getConfigDirectoryPath() + "/BaseViewer.ini";
    if(helper::system::FileSystem::exists(baseViewerFilename))
    {
        std::ifstream baseViewerStream(baseViewerFilename.c_str());
        for (std::string line; std::getline(baseViewerStream, line);)
        {
            const size_t equalPos = line.find('=');
            if( (equalPos != std::string::npos) && (equalPos != line.size()-1))
            {
                const std::string paramName = line.substr(0, equalPos);
                const bool booleanValue = line.substr(equalPos+1) == std::string("true") ;
                if(paramName == std::string("EnableSelectionDraw"))
                {
                    m_enableSelectionDraw = booleanValue;
                }
                else if(paramName == std::string("ShowSelectedNodeBoundingBox"))
                {
                    m_showSelectedNodeBoundingBox = booleanValue;
                }
                else if(paramName == std::string("ShowSelectedObjectBoundingBox"))
                {
                    m_showSelectedObjectBoundingBox = booleanValue;
                }
                else if(paramName == std::string("ShowSelectedObjectPositions"))
                {
                    m_showSelectedObjectPositions = booleanValue;
                }
                else if(paramName == std::string("ShowSelectedObjectSurfaces"))
                {
                    m_showSelectedObjectSurfaces = booleanValue;
                }
                else if(paramName == std::string("ShowSelectedObjectVolumes"))
                {
                    m_showSelectedObjectVolumes = booleanValue;
                }
                else if(paramName == std::string("ShowSelectedObjectIndices"))
                {
                    m_showSelectedObjectIndices = booleanValue;
                }
                else if(paramName == std::string("VisualScaling"))
                {
                    const float floatValue = std::stof(line.substr(equalPos+1)) ;
                    m_visualScaling = floatValue;
                }
            }
        }
    }


    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = sofa::core::objectmodel::New<sofa::component::visual::InteractiveCamera>();
            currentCamera->setName(groot->getNameHelper().resolveName(currentCamera->getClassName(), sofa::core::ComponentNameHelper::Convention::python));
            groot->addObject(currentCamera);
            //currentCamera->d_position.forceSet();
            //currentCamera->d_orientation.forceSet();
            currentCamera->bwdInit();
        }
        sofa::component::visual::VisualStyle::SPtr visualStyle = nullptr;
        groot->get(visualStyle);
        if (!visualStyle)
        {
            visualStyle = sofa::core::objectmodel::New<sofa::component::visual::VisualStyle>();
            visualStyle->setName(groot->getNameHelper().resolveName(visualStyle->getClassName(), sofa::core::ComponentNameHelper::Convention::python));

            core::visual::DisplayFlags* displayFlags = visualStyle->d_displayFlags.beginEdit();
            displayFlags->setShowVisualModels(sofa::core::visual::tristate::true_value);
            visualStyle->d_displayFlags.endEdit();

            groot->addObject(visualStyle);
            visualStyle->init();
        }

        currentCamera->setBoundingBox(groot->f_bbox.getValue().minBBox(), groot->f_bbox.getValue().maxBBox());

        // init pickHandler
        pick->init(groot.get());

        return true;
    }

    return false;
}

bool BaseViewer::unload()
{
    getPickHandler()->reset();
    getPickHandler()->unload();
    return true;
}

void BaseViewer::fitNodeBBox(sofa::core::objectmodel::BaseNode * node )
{
    if(!currentCamera) return;
    if( node->f_bbox.getValue().isValid() && !node->f_bbox.getValue().isFlat() )
        currentCamera->fitBoundingBox(
            node->f_bbox.getValue().minBBox(),
            node->f_bbox.getValue().maxBBox()
        );

    redraw();
}

void BaseViewer::fitObjectBBox(sofa::core::objectmodel::BaseObject * object)
{
    if(!currentCamera) return;

    if( object->f_bbox.getValue().isValid() && !object->f_bbox.getValue().isFlat() )
        currentCamera->fitBoundingBox(object->f_bbox.getValue().minBBox(),
                object->f_bbox.getValue().maxBBox());
    else
    {
        if(object->getContext()->f_bbox.getValue().isValid() && !object->getContext()->f_bbox.getValue().isFlat()  )
        {
            currentCamera->fitBoundingBox(
                object->getContext()->f_bbox.getValue().minBBox(),
                object->getContext()->f_bbox.getValue().maxBBox());
        }
    }
    redraw();
}

void BaseViewer::drawSelection(sofa::core::visual::VisualParams* vparams)
{
    if (!m_enableSelectionDraw)
        return;
    assert(vparams && "call of drawSelection without a valid visual param is not allowed");

    auto drawTool = vparams->drawTool();

    if(currentSelection.empty())
        return;

    drawTool->setPolygonMode(0, false);
    float screenHeight = vparams->viewport()[3];

    for(auto current : currentSelection)
    {
        using sofa::type::Vec3;
        using sofa::type::RGBAColor;
        using sofa::defaulttype::RigidCoord;
        using sofa::defaulttype::Rigid3Types;

        ////////////////////// Render when the selection is a Node ///////////////////////////////
        auto node = castTo<sofa::simulation::Node*>(current.get());
        if(node)
        {
            if(m_showSelectedNodeBoundingBox)
            {
                auto box = node->f_bbox.getValue();
                drawTool->drawBoundingBox(box.minBBox(), box.maxBBox(), 2.0);
            }

            // If it is a node then it is not a BaseObject, so we can continue.
            continue;
        }

        ////////////////////// Render when the selection is a BaseObject //////////////////////////
        auto object = castTo<sofa::core::objectmodel::BaseObject*>(current.get());
        if(object)
        {
            sofa::type::BoundingBox box;
            auto ownerNode = dynamic_cast<sofa::simulation::Node*>(object->getContext());
            if(ownerNode)
            {
                box = ownerNode->f_bbox.getValue();
            }

            if(m_showSelectedObjectBoundingBox)
            {
                drawTool->drawBoundingBox(box.minBBox(), box.maxBBox(), 2.0);
            }

            std::vector<Vec3> positions;
            auto position = object->findData("position");
            if(m_showSelectedObjectPositions)
            {
                if(position)
                {
                    auto positionsData = dynamic_cast<Data<sofa::type::vector<Vec3>>*>(position);
                    if(positionsData)
                    {
                        positions = positionsData->getValue();
                        drawTool->drawPoints(positions, 2.0, RGBAColor::yellow());
                    }
                    else
                    {
                        auto rigidPositions = dynamic_cast<Data<sofa::type::vector<RigidCoord<3, SReal>>>*>(position);
                        if(rigidPositions)
                        {
                            for(auto frame : rigidPositions->getValue())
                            {
                                float targetScreenSize = 50.0;
                                float distance = (currentCamera->getPosition() - Rigid3Types::getCPos(frame)).norm();
                                SReal scale = distance * tan(currentCamera->getFieldOfView() / 2.0f) * targetScreenSize / screenHeight;
                                drawTool->drawFrame(Rigid3Types::getCPos(frame), Rigid3Types::getCRot(frame), {scale, scale,scale});
                                positions.push_back(Rigid3Types::getCPos(frame));
                            }
                        }
                    }
                }
            }

            if(m_showSelectedObjectSurfaces && !positions.empty())
            {
                auto triangles = object->findData("triangles");
                if(triangles)
                {
                    auto d_triangles = dynamic_cast<Data<sofa::type::vector<core::topology::Topology::Triangle>>*>(triangles);
                    if(d_triangles)
                    {
                        std::vector<Vec3> tripoints;
                        for(auto indices : d_triangles->getValue())
                        {
                            if(indices[0] < positions.size() &&
                               indices[1] < positions.size() &&
                               indices[2] < positions.size())
                            {
                                tripoints.push_back(positions[indices[0]]);
                                tripoints.push_back(positions[indices[1]]);
                                tripoints.push_back(positions[indices[1]]);
                                tripoints.push_back(positions[indices[2]]);
                                tripoints.push_back(positions[indices[2]]);
                                tripoints.push_back(positions[indices[0]]);
                            }
                        }
                        drawTool->drawLines(tripoints, 1.5, RGBAColor::fromFloat(1.0,1.0,1.0,0.7));
                    }
                }
            }

            if(!positions.empty() && m_showSelectedObjectIndices)
            {
                const float scale = (box.maxBBox() - box.minBBox()).norm() * m_visualScaling;
                drawTool->draw3DText_Indices(positions, scale, RGBAColor::white());
            }

            continue;
        }
        msg_error("BaseViewer") << "Only node and object can be selected, if you see this line please report to sofa-developement team";
    }
}

void BaseViewer::setCurrentSelection(const std::set<sofa::core::objectmodel::Base::SPtr>& selection)
{
    currentSelection = selection;
}

const std::set<core::objectmodel::Base::SPtr> &BaseViewer::getCurrentSelection() const
{
    return currentSelection;
}

} // namespace sofa::gui::common
