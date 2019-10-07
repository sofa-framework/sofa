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

    ret->registerModel<DefaultObjectModel>();

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
      "UseDataDefinedColors": true
    }
  }
  )");
}



///////////////////////////////////////// ProfilerChartView ///////////////////////////////////

SofaWindowDataGraph::SofaWindowDataGraph(QWidget *parent, sofa::simulation::Node* scene)
    : QDialog(parent)
    , m_rootNode(scene)
    , m_scaleX(250)
    , m_scaleY(200)
{


    setConnecStyle();
    m_graphScene = new FlowScene(registerDataModels());
   
    m_exceptions = { "RequiredPlugin", "VisualStyle", "DefaultVisualManagerLoop", "InteractiveCamera" };
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    m_graphView = new FlowView(m_graphScene, this);
    layout->addWidget(m_graphView);
    this->setLayout(layout);
    
    resize(1000, 800);
    
    createComponentsNode();
}


void SofaWindowDataGraph::createComponentsNode()
{
 
    //parseChildrenNode(m_rootNode);
    std::cout << "SofaWindowDataGraph::createComponentsNode()" << std::endl;
    std::cout << "root: " << m_rootNode->getName() << std::endl;

    parseSimulationNode(m_rootNode, 0, 0);
    
}


void SofaWindowDataGraph::parseSimulationNode(sofa::simulation::Node* node, int posX, int posY)
{
    std::cout << posY << " ### Child Name: " << node->getName() << std::endl;
    // first parse the list BaseObject inside this node
    std::vector<sofa::core::objectmodel::BaseObject*> bObjects = node->getNodeObjects();
    int localPosX = posX;
    for (auto bObject : bObjects)
    {       
        bool skip = false;
        for (auto except : m_exceptions)
        {
            if (except == bObject->getClassName())
            {
                std::cout << "skip: " << except << std::endl;
                skip = true;
                break;
            }
        }
        
        if (skip)
            continue;
    
        addSimulationObject(bObject, localPosX, posY);
        localPosX++;
    }

    // second move to child nodes
    for (auto simuNode : node->getChildren())
    {
        posY++;
        posX++;
        parseSimulationNode(dynamic_cast<sofa::simulation::Node*>(simuNode), posX, posY);
    }
}

void SofaWindowDataGraph::addSimulationObject(sofa::core::objectmodel::BaseObject* bObject , int posX, int posY)
{
    const std::string& name = bObject->getClassName() + " - " + bObject->getName();
    std::cout << "addSimulationObject: " << name << std::endl;
    
    std::vector < std::pair < std::string, std::string> > data = filterUnnecessaryData(bObject);
    QtNodes::Node& fromNode = m_graphScene->createNode(std::make_unique<DefaultObjectModel>(data));
    fromNode.setObjectName(QString::fromStdString(bObject->getName()));
    
    DefaultObjectModel* model = dynamic_cast<DefaultObjectModel*>(fromNode.nodeDataModel());
    model->setCaption(name);

    auto& fromNgo = fromNode.nodeGraphicsObject();
    fromNgo.setPos(posX*m_scaleX, posY*m_scaleY);
}


std::vector < std::pair < std::string, std::string> > SofaWindowDataGraph::filterUnnecessaryData(sofa::core::objectmodel::BaseObject* bObject)
{
    helper::vector<sofa::core::objectmodel::BaseData*> allData = bObject->getDataFields();
    std::vector < std::pair < std::string, std::string> > filterData;
    for (auto data : allData)
    {
        const std::string& name = data->getName();
        const std::string& group = std::string(data->getGroup());

        if (data->getParent())
        { 
            sofa::core::objectmodel::BaseData* pData = data->getParent();
            std::cout << "- Parent: " << pData->getName() << " owwner: " << pData->getOwner()->getName() << std::endl;
            m_connections.push_back(DataGraphConnection(pData->getOwner()->getName(), pData->getName(), bObject->getName(), name));
        }
        

        if (name == "name" || name == "printLog" || name == "tags"
            || name == "bbox" || name == "listening")
            continue;

        if (group == "Visualization")
            continue;

        if (!group.empty())
        {
            std::cout << name << " -> " << data->getGroup() << std::endl;
        }
        filterData.push_back(std::pair<std::string, std::string>(name, data->getValueTypeString()));
    }
    std::cout << "## old Data: " << allData.size() << " - " << filterData.size() << std::endl;

    return filterData;
}


void SofaWindowDataGraph::connectNodeData()
{
    if (m_connections.empty())
        return;

    std::vector <QtNodes::Node*> nodes = m_graphScene->allNodes();

    for (auto connection : m_connections)
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

        
        DefaultObjectModel* modelP = dynamic_cast<DefaultObjectModel*>(parentNode->nodeDataModel());
        DefaultObjectModel* modelC = dynamic_cast<DefaultObjectModel*>(childNode->nodeDataModel());
        QtNodes::PortIndex parentId = modelP->getDataInputId(connection.m_parentDataName);
        QtNodes::PortIndex childId = modelC->getDataInputId(connection.m_childDataName);
        
        m_graphScene->createConnection(*childNode, childId, *parentNode, parentId);
    }
}


} // namespace qt

} // namespace gui

} // namespace sofa
