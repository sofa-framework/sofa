/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#ifndef GRAPHLISTENERQTreeWidget_H
#define GRAPHLISTENERQTreeWidget_H


#include "SofaGUIQt.h"

#include <QAbstractItemView>
#include <QTreeWidget>
#include <QTreeWidgetItem>


#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/MutationListener.h>



namespace sofa
{

namespace gui
{

namespace qt
{
using sofa::simulation::Node;
using sofa::simulation::Simulation;
using sofa::simulation::MutationListener;

QPixmap* getPixmap(core::objectmodel::Base* obj, bool, bool,bool);

class SOFA_SOFAGUIQT_API GraphListenerQListView : public MutationListener
{
public:
    //Q3ListView* widget;
    QTreeWidget* widget;
    bool frozen;
    std::map<core::objectmodel::Base*, QTreeWidgetItem* > items;
    std::map<core::objectmodel::BaseData*, QTreeWidgetItem* > datas;
    std::multimap<QTreeWidgetItem *, QTreeWidgetItem*> nodeWithMultipleParents;

    GraphListenerQListView(QTreeWidget* w)
        : widget(w), frozen(false)
    {
    }


    /*****************************************************************************************************************/
    QTreeWidgetItem* createItem(QTreeWidgetItem* parent);

    virtual void addChild(Node* parent, Node* child);
    virtual void removeChild(Node* parent, Node* child);
    virtual void moveChild(Node* previous, Node* parent, Node* child);
    virtual void addObject(Node* parent, core::objectmodel::BaseObject* object);
    virtual void removeObject(Node* /*parent*/, core::objectmodel::BaseObject* object);
    virtual void moveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object);
    virtual void addSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);
    virtual void removeSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);
    virtual void moveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);
    virtual void sleepChanged(Node* node);
    virtual void addDatas(core::objectmodel::BaseObject* parent);
    virtual void removeDatas(core::objectmodel::BaseObject* parent);
    virtual void freeze(Node* groot);
    virtual void unfreeze(Node* groot);
    core::objectmodel::Base* findObject(const QTreeWidgetItem* item);
    core::objectmodel::BaseData* findData(const QTreeWidgetItem* item);

};

}
}
}
#endif
