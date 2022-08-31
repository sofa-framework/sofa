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
#include <sofa/gui/qt/SofaSceneGraphWidget.h>

#include <map>

namespace sofa::gui::qt
{

class SOFA_GUI_QT_API QSofaListView : public SofaSceneGraphWidget
{
    Q_OBJECT
public:
    class LockContextManager
    {
    public:
        QSofaListView* self{nullptr};
        bool state{true};

        LockContextManager(QSofaListView* view, bool isLocked)
        {
            self = view;
            state = view->isLocked();
            if(isLocked)
                view->lock();
            else
                view->unLock();
        }

        ~LockContextManager()
        {
            if(state)
                self->lock();
            else
                self->unLock();
        }
    };

    QSofaListView(const SofaListViewAttribute& attribute,
            QWidget* parent = nullptr,
            const char* name = nullptr,
            Qt::WindowFlags f = Qt::WindowType::Widget );
    ~QSofaListView() override;

    GraphListenerQListView* getListener() const { return  graphListener_; }

    void setPropertyWidget(QDisplayPropertyWidget* propertyWid) {propertyWidget = propertyWid;}
    void addInPropertyWidget(QTreeWidgetItem *item, bool clear);

    void Clear(sofa::simulation::Node* rootNode);

    /// Updates the view so it is synchronized with the simulation graph.
    /// The view can be visually de-synchronized with the simulation graph. This happens
    /// when the view is "frozen" for performance reason. In that case, use isDirty to
    /// get current view state or the dirtynessChanged() signal.
    /// To resynchronize the view call the update methid.
    void update();
    void setRoot(sofa::simulation::Node*);

    SofaListViewAttribute getAttribute() const { return attribute_; }

    void contextMenuEvent(QContextMenuEvent *event) override;

    void expandPathFrom(const std::vector<std::string>& pathes);
    void getExpandedNodes(std::vector<std::string>&);

    void loadObject ( std::string path, double dx, double dy, double dz,  double rx, double ry, double rz,double scale ) = delete;

public Q_SLOTS:
    void Export();
    void CloseAllDialogs();
    void UpdateOpenedDialogs();
    void ExpandRootNodeOnly();

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
    void RemoveNode();
    void Modify();
    void openInEditor();
    void openInstanciation();
    void openImplementation();
    void copyFilePathToClipBoard();
    void DeactivateNode();
    void ActivateNode();
    void PutNodeToSleep();
    void WakeUpNode();

    void updateMatchingObjectmodel(QTreeWidgetItem* item, int);
    void updateMatchingObjectmodel(QTreeWidgetItem* item);

    void RunSofaRightClicked(const QPoint& point);
    void RunSofaDoubleClicked( QTreeWidgetItem* item, int index);

    void nodeNameModification( simulation::Node*);
    void focusObject();
    void focusNode();

protected:
    void expandPath(const std::string& path) ;
    void getExpandedNodes(QTreeWidgetItem* item, std::vector<std::string>&) ;
    void collapseNode(QTreeWidgetItem* item);
    void expandNode(QTreeWidgetItem* item);

    void transformObject ( sofa::simulation::Node *node, double dx, double dy, double dz,  double rx, double ry, double rz, double scale ) = delete;

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

} //namespace sofa::gui::qt
