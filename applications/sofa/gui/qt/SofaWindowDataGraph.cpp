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

    ret->registerModel<NaiveDataModel>();

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
   // FlowScene scene(registerDataModels());

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
        std::cout << localPosX << " -> " << bObject->getClassName() << "   " << std::endl;
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
    const std::string& name = bObject->getClassName() + "::" + bObject->getName();
    std::cout << "addSimulationObject: " << name << std::endl;
    QtNodes::Node& fromNode = m_graphScene->createNode(std::make_unique<NaiveDataModel>());
    
    NaiveDataModel* model = dynamic_cast<NaiveDataModel*>(fromNode.nodeDataModel());
    model->setCaption(name);

    auto& fromNgo = fromNode.nodeGraphicsObject();
   fromNgo.setPos(posX*m_scaleX, posY*m_scaleY);
}


} // namespace qt

} // namespace gui

} // namespace sofa
