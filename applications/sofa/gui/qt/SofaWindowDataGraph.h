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
#ifndef SOFA_WINDOW_DATAGRAPH
#define SOFA_WINDOW_DATAGRAPH

#include "PieWidget.h"
#include "QVisitorControlPanel.h"

#include <QTreeWidgetItem>
#include <QDrag>
#include <QPixmap>
#include <QTableWidget>
#include <QComboBox>
#include <sofa/gui/qt/SofaGuiQt.h>

#include <QDialog>
#include <QPainter>
#include <QTableWidget>

#include <iostream>
#include <deque>
#include <sofa/simulation/Node.h>


#define NODE_EDITOR_SHARED

#include <nodes/FlowView>

namespace sofa
{

namespace gui
{

namespace qt
{

class QtNodes::FlowScene;

class DataGraphConnection 
{
public:
    DataGraphConnection(const std::string& _pObjN, const std::string& _pDataN, 
        const std::string& _cObjN, const std::string& _cDataName)
        : m_parentObjName(_pObjN)
        , m_parentDataName(_pDataN)
        , m_childObjName(_cObjN)
        , m_childDataName(_cDataName)
    {}

    std::string m_parentObjName;
    std::string m_parentDataName;
    std::string m_childObjName;
    std::string m_childDataName;
};

class SofaWindowDataGraph : public QDialog
{
    Q_OBJECT
public:
    SofaWindowDataGraph(QWidget *parent, sofa::simulation::Node* scene);

   
protected:
    void createComponentsNode();

    void parseSimulationNode(sofa::simulation::Node* node, int posX, int posY);

    void addSimulationObject(sofa::core::objectmodel::BaseObject* bObject, int posX, int posY);

    std::vector < std::pair < std::string, std::string> > filterUnnecessaryData(sofa::core::objectmodel::BaseObject* bObject);
protected:
    QtNodes::FlowScene* m_graphScene;
    QtNodes::FlowView* m_graphView;

    sofa::simulation::Node* m_rootNode;
    SReal m_scaleX;
    SReal m_scaleY;

    std::vector<std::string> m_exceptions;
    std::vector<DataGraphConnection> m_connections;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_WINDOW_DATAGRAPH
