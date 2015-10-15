/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GRAPHVISITOR_H
#define SOFA_GRAPHVISITOR_H

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3TextDrag>
#include <QPixmap>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpixmap.h>
#endif

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/Node.h>

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

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif


class GraphVisitor
{
public:
    GraphVisitor(WindowVisitor *w) { window=w; graph=w->graphView; totalTimeMax=-1; initSize=false;}
    Q3ListViewItem *addNode(Q3ListViewItem *parent,Q3ListViewItem *elementAbove, std::string info);
    Q3ListViewItem *addComment(Q3ListViewItem *element, Q3ListViewItem *elementAbove, std::string comment);
    void addInformation(Q3ListViewItem *element, std::string name, std::string info);
    void addTime(Q3ListViewItem *element, std::string info);

    bool load(std::string &file);

    void setGraph(Q3ListView* g) {graph = g;}
    void clear() {graph->clear();}

    double getTotalTime(TiXmlNode* node) const;
    inline double getTime(TiXmlAttribute* attribute) const;

protected:
    void openTime           ( TiXmlNode* element, Q3ListViewItem* item);
    void openAttribute      ( TiXmlElement* element, Q3ListViewItem* item);
    Q3ListViewItem* openNode( TiXmlNode* node, Q3ListViewItem* parent, Q3ListViewItem* elementAbove);

    Q3ListView *graph;
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
