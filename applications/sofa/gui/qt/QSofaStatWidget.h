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
#ifndef SOFA_GUI_QT_QSOFASTATGRAPH_H
#define SOFA_GUI_QT_QSOFASTATGRAPH_H

#include <sofa/gui/qt/SofaGUIQt.h>
#include <sofa/helper/vector.h>

#ifdef SOFA_QT4
#include <QLabel>
#include <QWidget>
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3Header>
#else
#include <qlabel.h>
#include <qwidget.h>
#include <qlistview.h>
#include <qheader.h>
#endif

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
#endif

namespace sofa
{
namespace core
{
class CollisionModel;
namespace objectmodel
{
class Base;
}
}
namespace simulation
{
class Node;
}

namespace gui
{
namespace qt
{

class SOFA_SOFAGUIQT_API QSofaStatWidget : public QWidget
{
    Q_OBJECT
public:
    QSofaStatWidget(QWidget* parent);
    void CreateStats(sofa::simulation::Node* root);
protected:
    QLabel* statsLabel;
    Q3ListView* statsCounter;
    void addSummary();
    void addCollisionModelsStat(const sofa::helper::vector< sofa::core::CollisionModel* >& v);
    std::vector<std::pair<core::objectmodel::Base*, Q3ListViewItem*> > items_stats;

};
} //qt
} //gui
} //sofa

#endif //SOFA_GUI_QT_QSOFASTATGRAPH_H
