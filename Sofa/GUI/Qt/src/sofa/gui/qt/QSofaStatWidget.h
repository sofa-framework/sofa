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
#include <sofa/type/vector.h>

#include <QLabel>
#include <QWidget>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QListView>

namespace sofa::core
{
    class CollisionModel;
} // namespace sofa::core

namespace sofa::core::objectmodel
{
    class Base;
} // namespace sofa::core::objectmodel

namespace sofa::simulation
{
    class Node;
} // namespace sofa::core

namespace sofa::gui::qt
{

class SOFA_GUI_QT_API QSofaStatWidget : public QWidget
{
    Q_OBJECT
public:
    QSofaStatWidget(QWidget* parent);
    void CreateStats(sofa::simulation::Node* root);
protected:
    QLabel* statsLabel;
    QTreeWidget* statsCounter;
    void addSummary();
    void addCollisionModelsStat(const sofa::type::vector< sofa::core::CollisionModel* >& v);
    std::vector<std::pair<core::objectmodel::Base*, QTreeWidgetItem*> > items_stats;

};

} //namespace sofa::gui::qt
