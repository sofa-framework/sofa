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
#ifndef SOFA_GUI_QT_QSOFALISTVIEW_H
#define SOFA_GUI_QT_QSOFALISTVIEW_H

#include "SofaGUIQt.h"

#include <QWidget>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QPushButton>


#include <sofa/gui/qt/SofaGUIQt.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <map>

namespace sofa
{

namespace gui
{
namespace qt
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
} ObjectModel;

enum SofaListViewAttribute
{
    SIMULATION,
    VISUAL,
    MODELER
};

class SOFA_SOFAGUIQT_API QSofaListView : public QTreeWidget
{
    Q_OBJECT
public:
    QSofaListView(const SofaListViewAttribute& attribute,
            QWidget* parent=0,
            const char* name=0,
            Qt::WindowFlags f = 0 );
    ~QSofaListView();

    GraphListenerQListView* getListener() const { return  graphListener_; }

	void setPropertyWidget(QDisplayPropertyWidget* propertyWid) {propertyWidget = propertyWid;}
    void addInPropertyWidget(QTreeWidgetItem *item, bool clear);

    void Clear(sofa::simulation::Node* rootNode);
    void Freeze();
    void Unfreeze();
    SofaListViewAttribute getAttribute() const { return attribute_; }

	void contextMenuEvent(QContextMenuEvent *event);
public Q_SLOTS:
    void Export();
    void CloseAllDialogs();
    void UpdateOpenedDialogs();
Q_SIGNALS:
    void Close();
    void Lock(bool);
    void RequestSaving(sofa::simulation::Node*);
    void RequestExportOBJ(sofa::simulation::Node* node, bool exportMTL);
    void RequestActivation(sofa::simulation::Node*,bool);
	void RequestSleeping(sofa::simulation::Node*, bool);
    void RootNodeChanged(sofa::simulation::Node* newroot, const char* newpath);
    void NodeRemoved();
    void Updated();
    void NodeAdded();
    void focusChanged(sofa::core::objectmodel::BaseObject*);
    void focusChanged(sofa::core::objectmodel::BaseNode*);
    void dataModified( QString );

protected Q_SLOTS:
    void SaveNode();
    void exportOBJ();
    void collapseNode();
    void expandNode();
    void modifyUnlock(void* Id);
    void RaiseAddObject();
    void RemoveNode();
    void Modify();
    void HideDatas();
    void ShowDatas();
    void DeactivateNode();
    void ActivateNode();
	void PutNodeToSleep();
    void WakeUpNode();
    void loadObject ( std::string path, double dx, double dy, double dz,  double rx, double ry, double rz,double scale );

    void updateMatchingObjectmodel(QTreeWidgetItem* item, int);
    void updateMatchingObjectmodel(QTreeWidgetItem* item);

    void RunSofaRightClicked(const QPoint& point);
    void RunSofaDoubleClicked( QTreeWidgetItem* item, int index);

    void nodeNameModification( simulation::Node*);
    void focusObject();
    void focusNode();
protected:
    void collapseNode(QTreeWidgetItem* item);
    void expandNode(QTreeWidgetItem* item);
    void transformObject ( sofa::simulation::Node *node, double dx, double dy, double dz,  double rx, double ry, double rz, double scale );
    bool isNodeErasable( core::objectmodel::BaseNode* node);

    std::list<core::objectmodel::BaseNode*> collectNodesToChange(core::objectmodel::BaseNode* node);
    std::map< void*, QTreeWidgetItem* > map_modifyDialogOpened;
    std::map< void*, QDialog* > map_modifyObjectWindow;
    GraphListenerQListView* graphListener_;
    std::vector< std::string > list_object;
    AddObject* AddObjectDialog_;
    ObjectModel object_;
    SofaListViewAttribute attribute_;
	QDisplayPropertyWidget* propertyWidget;

};

} //sofa
} //gui
}//qt

#endif // SOFA_GUI_QT_QSOFALISTVIEW_H


