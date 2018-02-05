/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_GRAPHVISITOR_H
#define SOFA_GRAPHVISITOR_H

#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QDrag>
#include <QPixmap>

#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/Node.h>

#include "WindowVisitor.h"
#include "PieWidget.h"

#include <tinyxml.h>
#include <tinystr.h>

#include <iostream>
#include <set>

namespace sofa
{

namespace gui
{

namespace qt
{

class GraphVisitor
{
public:
    GraphVisitor(WindowVisitor *w) { window=w; graph=w->graphView; totalTimeMax=-1; initSize=false;}
    QTreeWidgetItem *addNode(QTreeWidgetItem *parent,QTreeWidgetItem *elementAbove, std::string info);
    QTreeWidgetItem *addComment(QTreeWidgetItem *element, QTreeWidgetItem *elementAbove, std::string comment);
    void addInformation(QTreeWidgetItem *element, std::string name, std::string info);
    void addTime(QTreeWidgetItem *element, std::string info);

    bool load(std::string &file);

    void setGraph(QTreeWidget* g) {graph = g;}
    void clear() {graph->clear();}

    double getTotalTime(TiXmlNode* node) const;
    inline double getTime(TiXmlAttribute* attribute) const;

protected:
    void openTime           ( TiXmlNode* element, QTreeWidgetItem* item);
    void openAttribute      ( TiXmlElement* element, QTreeWidgetItem* item);
    QTreeWidgetItem* openNode( TiXmlNode* node, QTreeWidgetItem* parent, QTreeWidgetItem* elementAbove);

    QTreeWidget *graph;
    WindowVisitor *window;

    double totalTime;
    double totalTimeMax;

    std::vector<double> timeComponentsBelow;
    int level;

    std::vector< dataTime > componentsTime;
    std::vector< dataTime > visitorsTime;

    std::vector< dataTime > componentsTimeTotal;
    std::vector< dataTime > visitorsTimeTotal;

    std::vector< dataTime > componentsTimeMax;
    std::vector< dataTime > visitorsTimeMax;

    std::vector< std::string > visitedNode;

    bool initSize;
};
}
}
}

#endif
