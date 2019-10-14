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
#include "SofaWindowDataGraph.h"

#include <QHeaderView>
#include <QMenu>
#include <QMessageBox>

#include <QGridLayout>
#include <QDebug>


#include <nodes/NodeData>
#include <nodes/FlowScene>
#include <nodes/DataModelRegistry>
#include <nodes/ConnectionStyle>
#include <nodes/Node>
#include "dataGraph/models.hpp"

namespace sofa
{

namespace gui
{

namespace qt
{
using namespace sofa::helper;


using QtNodes::DataModelRegistry;
using QtNodes::FlowScene;
using QtNodes::FlowView;
using QtNodes::ConnectionStyle;

static std::shared_ptr<DataModelRegistry>
registerDataModels()
{
    auto ret = std::make_shared<DataModelRegistry>();

    ret->registerModel<SofaComponentNodeModel>();

    /*
    We could have more models registered.
    All of them become items in the context meny of the scene.

    ret->registerModel<AnotherDataModel>();
    ret->registerModel<OneMoreDataModel>();

    */

    return ret;
}


static
void
setConnecStyle()
{
    ConnectionStyle::setConnectionStyle(
        R"(
  {
    "ConnectionStyle": {
      "LineWidth": 3.0,
      "UseDataDefinedColors": true
    }
  }
  )");
}



///////////////////////////////////////// ProfilerChartView ///////////////////////////////////

SofaWindowDataGraph::SofaWindowDataGraph(QWidget *parent, sofa::simulation::Node* scene)
    : QDialog(parent)
    , m_rootNode(scene)
    , m_scaleX(10)
    , m_scaleY(30)
    , m_posX(0)
    , m_posY(0)
    , debugNodeGraph(false)
{
    setConnecStyle();
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::WindowMaximizeButtonHint;
    flags |= Qt::WindowContextHelpButtonHint;
    setWindowFlags(flags);

    m_graphScene = new FlowScene(registerDataModels());
   
    m_exceptions = { "RequiredPlugin", "VisualStyle", "DefaultVisualManagerLoop", "InteractiveCamera" };
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    m_graphView = new FlowView(m_graphScene, this);
    layout->addWidget(m_graphView);
    this->setLayout(layout);
    
    resize(1000, 800);
    
    createComponentsNode();

    connectNodeData();

    m_graphView->scaleDown();
}


SofaWindowDataGraph::~SofaWindowDataGraph()
{
    clearNodeData();
    // todo check if m_graphView need to be deleted. Normally no as child of QtWidget RealGui.
    //delete m_graphView;
}

void SofaWindowDataGraph::clearNodeData()
{
    m_dataLinks.clear();

    if (m_graphScene != nullptr)
    {
        msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "clear before: " << m_graphScene->allNodes().size();
        //m_graphScene->clear();
        delete m_graphScene;
        m_graphScene = new FlowScene(registerDataModels());
        m_graphView->setScene(m_graphScene);
        
        msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "clear after: " << m_graphScene->allNodes().size();
    }
    m_posX = 0;
    m_posY = 0;
}

void SofaWindowDataGraph::resetNodeGraph(sofa::simulation::Node* scene)
{
    m_rootNode = scene;
    clearNodeData();

    createComponentsNode();

    connectNodeData();
}

void SofaWindowDataGraph::createComponentsNode()
{
    //parse Children Node starting from root
    parseSimulationNode(m_rootNode);
    
}


void SofaWindowDataGraph::parseSimulationNode(sofa::simulation::Node* node, int posX)
{
    msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << m_posY << " ### Child Name: " << node->getName();
    // first parse the list BaseObject inside this node
    std::vector<sofa::core::objectmodel::BaseObject*> bObjects = node->getNodeObjects();
    m_posX = posX;
    int maxData = 0;
    for (auto bObject : bObjects)
    {       
        bool skip = false;
        for (auto except : m_exceptions)
        {
            if (except == bObject->getClassName())
            {
                msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "skip: " << except;
                skip = true;
                break;
            }
        }
        
        if (skip)
            continue;
    
        size_t nbrData = addSimulationObject(bObject);
        if (nbrData > maxData)
            maxData = nbrData;

        // space between cells
        m_posX += 14 * m_scaleX;
    }

    if (bObjects.size() >= 4) {
        m_posY += (maxData + 10) * m_scaleY;
        m_posX = posX + 30 * m_scaleX;
    }

    // second move to child nodes
    for (auto simuNode : node->getChildren())
    {
        parseSimulationNode(dynamic_cast<sofa::simulation::Node*>(simuNode), m_posX);
    }
}

size_t SofaWindowDataGraph::addSimulationObject(sofa::core::objectmodel::BaseObject* bObject)
{
    const std::string& name = bObject->getClassName() + " - " + bObject->getName();
    msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "addSimulationObject: " << name;
    
    std::vector < std::pair < std::string, std::string> > data = filterUnnecessaryData(bObject);
    QtNodes::Node& fromNode = m_graphScene->createNode(std::make_unique<SofaComponentNodeModel>(data));
    fromNode.setObjectName(QString::fromStdString(bObject->getName()));
    
    SofaComponentNodeModel* model = dynamic_cast<SofaComponentNodeModel*>(fromNode.nodeDataModel());
    model->setCaption(name);

    auto& fromNgo = fromNode.nodeGraphicsObject();    
    fromNgo.setPos(m_posX, m_posY);
    m_posX += name.length() * m_scaleX;

    return data.size();
}


std::vector < std::pair < std::string, std::string> > SofaWindowDataGraph::filterUnnecessaryData(sofa::core::objectmodel::BaseObject* bObject)
{    
    std::vector < std::pair < std::string, std::string> > filterData;
    // first add this object name as first Data (to be used for the links representation)
    filterData.push_back(std::pair<std::string, std::string>(bObject->getName(), "name"));

    // parse links
    const sofa::core::objectmodel::Base::VecLink& links = bObject->getLinks();
    for (auto link : links)
    {
        const std::string& name = link->getName();
        // ignore unnamed link
        if (link->getName().empty())
            continue;

        // ignore link to context
        if (link->getName() == "context")
            continue;

        if (!link->storePath() && 0 == link->getSize())
            continue;

        const std::string valuetype = link->getValueTypeString();

        msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "## link: " << name << " | link->getSize(): " << link->getSize() << " | valuetype: " << valuetype << " | path: " << link->storePath();
        

        std::string linkPath = link->getLinkedPath();
        linkPath.erase(0, 1); // remove @
        std::size_t found = linkPath.find_last_of("/");
        if (found != std::string::npos) // remove path
            linkPath.erase(0, found);

        msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "  # baselink: " << linkPath;
        m_dataLinks.push_back(DataGraphConnection(linkPath, linkPath, bObject->getName(), bObject->getName()));
    }

    // parse all Data
    helper::vector<sofa::core::objectmodel::BaseData*> allData = bObject->getDataFields();    
    for (auto data : allData)
    {
        const std::string& name = data->getName();
        const std::string& group = std::string(data->getGroup());

        if (data->getParent())
        { 
            sofa::core::objectmodel::BaseData* pData = data->getParent();
            msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "- Parent: " << pData->getName() << " owwner: " << pData->getOwner()->getName();
            m_dataLinks.push_back(DataGraphConnection(pData->getOwner()->getName(), pData->getName(), bObject->getName(), name));
        }
        

        if (name == "name" || name == "printLog" || name == "tags"
            || name == "bbox" || name == "listening")
            continue;

        if (group == "Visualization")
            continue;

        if (!group.empty())
        {
            msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << name << " -> " << data->getGroup();
        }
        filterData.push_back(std::pair<std::string, std::string>(name, data->getValueTypeString()));
    }

    //msg_info_when(debugNodeGraph, "SofaWindowDataGraph") << "## old Data: " << allData.size() << " - " << filterData.size();

    return filterData;
}


void SofaWindowDataGraph::connectNodeData()
{
    if (m_dataLinks.empty())
        return;

    std::vector <QtNodes::Node*> nodes = m_graphScene->allNodes();

    for (auto connection : m_dataLinks)
    {
        QtNodes::Node* parentNode = nullptr;
        QtNodes::Node* childNode = nullptr;
        int cpt = 0;

        for (unsigned int i = 0; i < nodes.size(); ++i)
        {
            std::string objName = nodes[i]->objectName().toStdString();
            if (parentNode == nullptr && objName == connection.m_parentObjName)
            {
                parentNode = nodes[i];
                cpt++;
            }

            if (childNode == nullptr && objName == connection.m_childObjName)
            {
                childNode = nodes[i];
                cpt++;
            }

            if (cpt == 2)
                break;
        }

        if (cpt != 2)
        {
            msg_error("SofaWindowDataGraph") << "Object not found while creating connection between " << connection.m_parentObjName << " and child: " << connection.m_childObjName;
            continue;
        }

        
        SofaComponentNodeModel* modelP = dynamic_cast<SofaComponentNodeModel*>(parentNode->nodeDataModel());
        SofaComponentNodeModel* modelC = dynamic_cast<SofaComponentNodeModel*>(childNode->nodeDataModel());
        QtNodes::PortIndex parentId = modelP->getDataInputId(connection.m_parentDataName);
        QtNodes::PortIndex childId = modelC->getDataInputId(connection.m_childDataName);
        
        m_graphScene->createConnection(*childNode, childId, *parentNode, parentId);
    }
}

void SofaWindowDataGraph::connectNodeLinks()
{

}


} // namespace qt

} // namespace gui

} // namespace sofa
