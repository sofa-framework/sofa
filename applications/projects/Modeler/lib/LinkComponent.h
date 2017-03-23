/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_LINKCOMPONENT_H
#define SOFA_LINKCOMPONENT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/loader/BaseLoader.h>

#include "ModifierCondition.h"
#include "GraphModeler.h"

#include <QComboBox>
#include <QLineEdit>
#include <QStringList>
#include <QCheckBox>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QMenu>


namespace sofa
{

namespace gui
{

namespace qt
{
/// Qt Widget applying linking between a component and a loader
class LinkComponent : public QWidget
{
    Q_OBJECT
public:
    typedef helper::vector< sofa::core::loader::BaseLoader* > Loaders;
    LinkComponent(GraphModeler* mg, const std::vector<QTreeWidgetItem*>& items, QTreeWidgetItem* sel);
    ~LinkComponent();

    // get the number of loader in the tree branch
    unsigned int loaderNumber() const;

public slots:
    void applyLinkComponent();
signals:
    void displayMessage(const std::string &message);
protected:
    QTreeWidget* listView;
    GraphModeler* mainGraph;
    std::map<QTreeWidgetItem*, sofa::core::loader::BaseLoader*> items2components;
    QTreeWidgetItem* selectedComponent;
    unsigned int loaderNum;
};

}
}
}

#endif // SOFA_LINKCOMPONENT_H
