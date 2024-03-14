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

#include <QAbstractItemView>
#include <QTreeWidget>
#include <QTreeWidgetItem>

#include <sofa/simulation/fwd.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/simulation/MutationListener.h>
#include <sofa/gui/qt/QSofaListView.h>

namespace sofa::gui::qt
{
using sofa::simulation::Node;
using sofa::simulation::MutationListener;

QPixmap* getPixmap(core::objectmodel::Base* obj, bool, bool,bool);


/// A listener to connect changes on the component state with its graphical view.
/// The listener is added to the ComponentState of an object to track changes to
/// and update the icon/treewidgetitem when this happens.
class ObjectStateListener : public sofa::core::objectmodel::DDGNode
{
public:
    QTreeWidgetItem* item;

    // Use a SPtr here because otherwise sofa may decide to remove the base without notifying the ObjectStateListener
    // is going to a segfault the right way.
    sofa::core::objectmodel::Base::SPtr object;

    ObjectStateListener(QTreeWidgetItem* item_, sofa::core::objectmodel::Base* object_);
    ~ObjectStateListener() override;
    void update() override;
    void notifyEndEdit() override;
};


class SOFA_GUI_QT_API GraphListenerQListView : public MutationListener
{
public:
    SofaSceneGraphWidget* widget;
    std::map<core::objectmodel::Base*, ObjectStateListener* > listeners;
    std::map<core::objectmodel::Base*, QTreeWidgetItem* > items;
    std::map<core::objectmodel::BaseData*, QTreeWidgetItem* > datas;
    std::multimap<QTreeWidgetItem *, QTreeWidgetItem*> nodeWithMultipleParents;

    GraphListenerQListView(SofaSceneGraphWidget* w)
        : widget(w)
    {
    }
    ~GraphListenerQListView() override;

    /*****************************************************************************************************************/
    QTreeWidgetItem* createItem(QTreeWidgetItem* parent);
    virtual void onBeginAddChild(Node* parent, Node* child) override;
    virtual void onBeginRemoveChild(Node* parent, Node* child) override;
    virtual void onBeginAddObject(Node* parent, core::objectmodel::BaseObject* object) override;
    virtual void onBeginRemoveObject(Node* /*parent*/, core::objectmodel::BaseObject* object) override;
    virtual void onBeginAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) override;
    virtual void onBeginRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) override;

    virtual void sleepChanged(Node* node) override;
    virtual void addDatas(core::objectmodel::BaseObject* parent);
    virtual void removeDatas(core::objectmodel::BaseObject* parent);
    core::objectmodel::Base* findObject(const QTreeWidgetItem* item);
    core::objectmodel::BaseData* findData(const QTreeWidgetItem* item);

    inline static QColor nameColor { 120, 120, 120};
};

} // namespace sofa::gui::qt
