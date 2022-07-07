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

#include <QWidget>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QPushButton>


#include <sofa/gui/qt/config.h>
#include <sofa/simulation/fwd.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <map>

namespace sofa::gui::qt
{

class AddObject;
class GraphListenerQListView;
class QDisplayPropertyWidget;

enum ObjectModelType { typeNode, typeObject, typeData };
typedef union ObjectModelPtr
{
    sofa::simulation::Node* Node;
    core::objectmodel::BaseObject* Object;
    core::objectmodel::BaseData* Data;
} ObjectModelPtr;

typedef struct ObjectModel
{
public:
    ObjectModelType type;
    ObjectModelPtr ptr;
    bool isNode()   { return type == typeNode;   }
    bool isObject() { return type == typeObject; }
    bool isData()   { return type == typeData;   }
    bool isBase()   { return isNode() || isObject(); }
    sofa::core::objectmodel::Base* asBase()
    {
        if( isNode() )
            return sofa::core::castToBase(ptr.Node);
        if( isObject() )
            return dynamic_cast<sofa::core::objectmodel::Base*>(ptr.Object);
        return nullptr;
    }
} ObjectModel;

enum SofaListViewAttribute
{
    SIMULATION,
    VISUAL,
    MODELER
};

class SOFA_GUI_QT_API SofaSceneGraphWidget : public QTreeWidget
{
    Q_OBJECT
public:
    SofaSceneGraphWidget(QWidget* parent) : QTreeWidget(parent){}
    ~SofaSceneGraphWidget(){}

    void lock();
    void unLock();

    /// Returns true if the view is not syncrhonized anymore with the simulation graph.
    /// To re-syncronize the view you can:
    ///     - call unfreeze() so any future change will be reflected
    ///     - call update(), to update one time the graph.
    bool isDirty();

    /// Returns true if the view updates for any scene graph change is disable.
    bool isLocked();

    /// call this method to indicate that the internal model has changed
    /// and thus the view is now dirty.
    void setViewToDirty();

Q_SIGNALS:
    /// Connect to this signal to be notified when the dirtyness status of the QSofaListView changed.
    void dirtynessChanged(bool isDirty);

    /// Connect to this signal to be notified when the locking status changed
    void lockingChanged(bool isLocked);

protected:
    /// Indicate that the view is de-synchronized with the real content of the simulation graph.
    /// This can happen if the graph has been freezed (i.e. not graphically updated) for performance
    /// reason while simulating complex scenes.
    bool m_isDirty;
    bool m_isLocked;
};

} //namespace sofa::gui::qt
