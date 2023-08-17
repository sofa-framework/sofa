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
#include "QSofaListView.h"
#include "QDisplayPropertyWidget.h"
#include "GraphListenerQListView.h"
#include "ModifyObject.h"
#include "GenGraphForm.h"
#include "RealGUI.h"
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/common/TransformationVisitor.h>
#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/simulation/common/xml/XML.h>
#include <sofa/helper/cast.h>
#include <QMenu>
#include <QtGlobal> // version macro
#include <QMessageBox>
#include <QApplication>

#include <QDesktopServices>
#include <QFileInfo>
#include <QUrl>
#include <QClipboard>
#include <QSettings>

using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using namespace sofa::gui::common;

namespace sofa::gui::qt
{

QSofaListView::QSofaListView(const SofaListViewAttribute& attribute,
                             QWidget* parent,
                             const char* name,
                             Qt::WindowFlags f):
    SofaSceneGraphWidget(parent),
    graphListener_(nullptr),
    AddObjectDialog_(nullptr),
    attribute_(attribute),
    propertyWidget(nullptr)
{
    this->setObjectName(name);
    this->setWindowFlags(f);
    //List of objects
    //Read the object.txt that contains the information about the objects which can be added to the scenes whithin a given BoundingBox and scale range
    std::string object ( "config/object.txt" );

    if( sofa::helper::system::DataRepository.findFile ( object ) )
    {
        list_object.clear();
        std::ifstream end(object.c_str());
        std::string s;
        while( end >> s )
        {
            list_object.push_back(s);
        }
        end.close();
    }

    this->setColumnCount(2);
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
    header()->setResizeMode(0, QHeaderView::Interactive);
    header()->setResizeMode(1, QHeaderView::Stretch);
#else
    header()->setSectionResizeMode(0, QHeaderView::Interactive);
    header()->setSectionResizeMode(1, QHeaderView::Stretch);
#endif // SOFA_QT5
    QStringList headerLabels;
    headerLabels << "Name" << "Class";
    this->setHeaderLabels(headerLabels);

    setRootIsDecorated(true);
    setIndentation(8);

    graphListener_ = new GraphListenerQListView(this);

    this->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &QSofaListView::customContextMenuRequested ,this, &QSofaListView::RunSofaRightClicked);
    connect(this, &QSofaListView::itemDoubleClicked, this, &QSofaListView::RunSofaDoubleClicked);
    connect(this, &QSofaListView::itemClicked, this, [&](QTreeWidgetItem *item, int){ updateMatchingObjectmodel(item); });
}

QSofaListView::~QSofaListView()
{
    delete graphListener_;
}


void QSofaListView::Clear(Node* /*rootNode*/)
{
    /*
    if(graphListener_ != nullptr)
    {
        delete graphListener_;
    }

    CloseAllDialogs();
    clear();
    graphListener_ = new GraphListenerQListView(this);

    this->setSortingEnabled(false);

    rootNode->addListener(graphListener_);
    update();

    std::map<Base*, QTreeWidgetItem* >::iterator graph_iterator;
    for (graph_iterator = graphListener_->items.begin();
         graph_iterator != graphListener_->items.end();
         ++graph_iterator)
    {
        Node* node = dynamic_cast< Node* >(graph_iterator->first);
        if (node!=nullptr && !node->isActive())
        {
            object_.ptr.Node = node;
            object_.type  = typeNode;
            emit RequestActivation(object_.ptr.Node, node->isActive());
        }
    }
    */
}

void QSofaListView::CloseAllDialogs()
{
    emit Close();
    assert( map_modifyObjectWindow.empty() );
    assert( map_modifyDialogOpened.empty() );

}

void QSofaListView::modifyUnlock(void* Id)
{
    map_modifyDialogOpened.erase( Id );
    map_modifyObjectWindow.erase( Id );
}

/// Traverse the item tree and retrive the item that are expanded. The path of the node
/// that are expanded are stored in the pathes std::vector::std::string>.
void QSofaListView::getExpandedNodes(QTreeWidgetItem* item, std::vector<std::string>& pathes)
{
    if(!item)
        return;

    /// We have reached a leaf of the hierarchy or it is closed...so we save the path
    if( !item->isExpanded() && graphListener_->findObject(item)->toBaseNode() != nullptr )
        return;

    const BaseNode* parentNode = graphListener_->findObject(item)->toBaseNode() ;
    if(parentNode == nullptr)
        return;

    const std::string path = parentNode->getPathName();
    pathes.push_back(path);

    for(int i=0 ; i<item->childCount() ; i++)
    {
        QTreeWidgetItem* child = item->child(i);
        const BaseNode* childNode = graphListener_->findObject(child)->toBaseNode() ;

        if(childNode==nullptr)
            continue;

        if( childNode->getParents()[0] == parentNode )
            getExpandedNodes(child, pathes) ;
    }

    return ;
}

void QSofaListView::getExpandedNodes(std::vector<std::string>& pathes)
{
    LockContextManager lock(this, true);
    QTreeWidgetItem* rootitem = this->topLevelItem(0) ;
    getExpandedNodes(rootitem,pathes) ;
}

void QSofaListView::collapseNode()
{
    collapseNode(currentItem());
}

void QSofaListView::collapseNode(QTreeWidgetItem* item)
{
    if (!item) return;

    LockContextManager lock(this, true);
    for(int i=0 ; i<item->childCount() ; i++)
    {
        QTreeWidgetItem* child = item->child(i);
        child->setExpanded(false);
    }
    item->setExpanded ( true );
}

void QSofaListView::expandPath(const std::string& path)
{
    if(path.empty())
        return;

    if(path.data()[0] != '/')
        return;

    Node* match = down_cast<Node>( graphListener_->findObject(this->topLevelItem(0))->toBaseNode() );

    QStringList tokens = QString::fromStdString(path).split('/') ;

    for(int i=1;i<tokens.size();i++)
    {
        match = match->getChild(tokens[i].toStdString());

        if(match == nullptr)
            return;

        if(graphListener_->items.find(match) != graphListener_->items.end())
        {
            QTreeWidgetItem* item = graphListener_->items[match] ;
            item->setExpanded ( true );
        }
    }
}

void QSofaListView::expandPathFrom(const std::vector<std::string>& pathes)
{
    LockContextManager lock(this, true);
    for(auto& path : pathes)
    {
        expandPath(path) ;
    }
}


void QSofaListView::expandNode()
{
    expandNode(currentItem());
}

void QSofaListView::expandNode(QTreeWidgetItem* item)
{
    if (!item)
        return;

    LockContextManager lock(this, true);
    item->setExpanded ( true );

    for(int i=0 ; i<item->childCount() ; i++)
    {
        QTreeWidgetItem* child = item->child(i);
        child->setExpanded(true);
        expandNode(child);
    }

}

void QSofaListView::setRoot(Node* root)
{
    if(!root)
        return;

    CloseAllDialogs();
    clear();

    if(graphListener_)
        delete graphListener_;
    graphListener_ = new GraphListenerQListView(this);

    setSortingEnabled(false);

    const bool lockStatus = m_isLocked;
    m_isLocked=false;
    root->addListener(graphListener_);
    graphListener_->onBeginAddChild(nullptr, root);
    m_isLocked=lockStatus;
    m_isDirty=false;

    emit dirtynessChanged(m_isDirty);
}

void QSofaListView::update()
{
    if(!m_isDirty)
        return;

    m_isDirty=false;

    if(!graphListener_ || !this->topLevelItem(0))
    {
        emit dirtynessChanged(m_isDirty);
        return;
    }
    emit dirtynessChanged(m_isDirty);
}

void QSofaListView::updateMatchingObjectmodel(QTreeWidgetItem* item, int)
{
    updateMatchingObjectmodel(item);
}

void QSofaListView::updateMatchingObjectmodel(QTreeWidgetItem* item)
{
    BaseData* data = nullptr;
    Base* base = nullptr;
    BaseObject* object = nullptr;
    BaseNode* basenode = nullptr;
    if(item == nullptr)
    {
        object_.ptr.Node = nullptr;
    }
    else
    {
        base = graphListener_->findObject(item);
        if(base == nullptr)
        {
            data = graphListener_->findData(item);
            assert(data);
            object_.ptr.Data = data;
            object_.type = typeData;
            return;
        }
        basenode = base->toBaseNode();
        if( basenode == nullptr)
        {
            object = dynamic_cast<BaseObject*>(base);
            object_.ptr.Object = object;
            object_.type = typeObject;
        }
        else
        {
            object_.ptr.Node = down_cast<Node>(basenode);
            object_.type = typeNode;
        }
    }

    addInPropertyWidget(item, true);
}

void QSofaListView::addInPropertyWidget(QTreeWidgetItem *item, bool clear)
{
    if(!item)
        return;

    Base* object = graphListener_->findObject(item);
    if(object == nullptr)
        return;

    if(propertyWidget)
    {
        propertyWidget->addComponent(object->getName().c_str(), object, item, clear);

        propertyWidget->show();
    }
}

void QSofaListView::contextMenuEvent(QContextMenuEvent *event)
{
    event->accept();
}

void QSofaListView::focusObject()
{
    if( object_.isObject())
        emit( focusChanged(object_.ptr.Object));

}

void QSofaListView::focusNode()
{
    if( object_.isNode())
        emit( focusChanged(object_.ptr.Node));
}

/*****************************************************************************************************************/
void QSofaListView::RunSofaRightClicked( const QPoint& point)
{
    QTreeWidgetItem *item = this->itemAt( point );

    if( item == nullptr) return;

    updateMatchingObjectmodel(item);

    QAction* act;
    bool object_hasData = false;
    if(object_.type == typeObject)
    {
        object_hasData = object_.ptr.Object->getDataFields().size() > 0 ? true : false;
    }
    QMenu *contextMenu = new QMenu ( this );
    contextMenu->setObjectName( "ContextMenu");
    if( object_.isNode() )
    {
        act = contextMenu->addAction("Focus", this,SLOT(focusNode()));
        const bool enable = object_.ptr.Node->f_bbox.getValue().isValid() && !object_.ptr.Node->f_bbox.getValue().isFlat();
        act->setEnabled(enable);
    }
    if( object_.isObject() )
    {
        act = contextMenu->addAction("Focus", this,SLOT(focusObject()));
        const bool enable = object_.ptr.Object->f_bbox.getValue().isValid() && !object_.ptr.Object->f_bbox.getValue().isFlat() ;
        act->setEnabled(enable);
    }

    contextMenu->addSeparator();

    //Creation of the context Menu
    if ( object_.type == typeNode)
    {
        act = contextMenu->addAction("Collapse", this,SLOT(collapseNode()));
        act = contextMenu->addAction("Expand", this,SLOT(expandNode()));
        contextMenu->addSeparator();
        /*****************************************************************************************************************/
        if (object_.ptr.Node->isActive())
        {
            act = contextMenu->addAction("Deactivate", this,SLOT(DeactivateNode()));
        }
        else
        {
            act = contextMenu->addAction("Activate", this,SLOT(ActivateNode()));
        }
        if (object_.ptr.Node->isSleeping())
        {
            act = contextMenu->addAction("Wake up", this,SLOT(WakeUpNode()));
        }
        else
        {
            act = contextMenu->addAction("Put to sleep", this,SLOT(PutNodeToSleep()));
        }
        contextMenu->addSeparator();
        /*****************************************************************************************************************/
        act = contextMenu->addAction("Save Node", this,SLOT(SaveNode()));
        act = contextMenu->addAction("Export OBJ", this,SLOT(exportOBJ()));

        if ( attribute_ == SIMULATION)
        {
            act = contextMenu->addAction("Remove Node", this,SLOT(RemoveNode()));
            //If one of the elements or child of the current node is beeing modified, you cannot allow the user to erase the node
            if ( !isNodeErasable ( object_.ptr.Node ) )
            {
                act->setEnabled(false);
            }
        }
    }
    act = contextMenu->addAction("Modify", this,SLOT(Modify()));

    if( object_.isBase() )
    {
        contextMenu->addSeparator();
        act = contextMenu->addAction("Go to Scene...", this, SLOT(openInstanciation()));
        act->setEnabled(object_.asBase()->getInstanciationSourceFileName() != "");
        act = contextMenu->addAction("Go to Implementation...", this, SLOT(openImplementation()));
        act->setEnabled(object_.asBase()->getDefinitionSourceFileName() != "");
    }

    contextMenu->addSeparator();
    act = contextMenu->addAction("Copy file path", this,SLOT(copyFilePathToClipBoard()));
    act = contextMenu->addAction("Open file in editor", this,SLOT(openInEditor()));

    contextMenu->exec ( this->mapToGlobal(point) /*, index */);
}

void QSofaListView::RunSofaDoubleClicked(QTreeWidgetItem* item, int /*index*/)
{
    if(item == nullptr)
    {
        return;
    }

    item->setExpanded( !item->isExpanded());
    Modify();
}

/*****************************************************************************************************************/
void QSofaListView::nodeNameModification(simulation::Node* node)
{
    QTreeWidgetItem *item=graphListener_->items[node];

    const QString nameToUse(node->getName().c_str());
    item->setText(0,nameToUse);

    typedef std::multimap<QTreeWidgetItem *, QTreeWidgetItem*>::iterator ItemIterator;
    const std::pair<ItemIterator,ItemIterator> range=graphListener_->nodeWithMultipleParents.equal_range(item);

    for (ItemIterator it=range.first; it!=range.second; ++it) it->second->setText(0,nameToUse);
}


void QSofaListView::DeactivateNode()
{
    emit RequestActivation(object_.ptr.Node,false);
    currentItem()->setExpanded(false);

}

void QSofaListView::ActivateNode()
{
    emit RequestActivation(object_.ptr.Node,true);
}

void QSofaListView::PutNodeToSleep()
{
    emit RequestSleeping(object_.ptr.Node, true);
}

void QSofaListView::WakeUpNode()
{
    emit RequestSleeping(object_.ptr.Node, false);
}

void QSofaListView::SaveNode()
{
    if( object_.ptr.Node != nullptr)
    {
        LockContextManager lock(this, true);
        Node * node = object_.ptr.Node;
        emit RequestSaving(node);
    }
}
void QSofaListView::exportOBJ()
{
    if( object_.ptr.Node != nullptr)
    {
        LockContextManager lock(this, true);
        Node * node = object_.ptr.Node;
        emit RequestExportOBJ(node,true);
    }
}

void QSofaListView::RemoveNode()
{
    if( object_.type == typeNode)
    {
        LockContextManager lock(this, true);
        const Node::SPtr node = object_.ptr.Node;
        if ( node == node->getRoot() )
        {
            if ( QMessageBox::warning ( this, "Removing root", "root node cannot be removed" ) )
                return;
        }
        else
        {
            node->detachFromGraph();
            node->execute<simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
            emit NodeRemoved();
        }
    }
}

void QSofaListView::Modify()
{
    void *current_Id_modifyDialog = nullptr;
    LockContextManager lock(this, true);
    if ( currentItem() != nullptr )
    {
        ModifyObjectFlags dialogFlags = ModifyObjectFlags();
        dialogFlags.setFlagsForSofa();
        ModifyObject* dialogModifyObject = nullptr;

        if (object_.type == typeData)       //user clicked on a data
        {
            current_Id_modifyDialog = object_.ptr.Data;
        }
        if (object_.type == typeNode)
        {
            current_Id_modifyDialog = object_.ptr.Node;
        }
        if(object_.type == typeObject)
        {
            current_Id_modifyDialog = object_.ptr.Object;
        }
        assert(current_Id_modifyDialog != nullptr);

        //Opening of a dialog window automatically created

        const std::map< void*, QDialog* >::iterator testWindow =  map_modifyObjectWindow.find( current_Id_modifyDialog);
        if ( testWindow != map_modifyObjectWindow.end())
        {
            //Object already being modified: no need to open a new window
            (*testWindow).second->raise();
            return;
        }

        dialogModifyObject = new ModifyObject(current_Id_modifyDialog,currentItem(),this,dialogFlags,currentItem()->text(0).toStdString().c_str());
        if(object_.type == typeData)
            dialogModifyObject->createDialog(object_.ptr.Data);
        if(object_.type == typeNode)
            dialogModifyObject->createDialog(dynamic_cast<Base*>(object_.ptr.Node));
        if(object_.type  == typeObject)
            dialogModifyObject->createDialog(dynamic_cast<Base*>(object_.ptr.Object));

        map_modifyDialogOpened.insert( std::make_pair ( current_Id_modifyDialog, currentItem()) );
        map_modifyObjectWindow.insert( std::make_pair(current_Id_modifyDialog, dialogModifyObject));
        connect ( dialogModifyObject, &ModifyObject::objectUpdated, this, &QSofaListView::Updated );
        connect ( this, &QSofaListView::Close, dialogModifyObject, &ModifyObject::closeNow );
        connect ( dialogModifyObject, &ModifyObject::dialogClosed, this, &QSofaListView::modifyUnlock );
        connect ( dialogModifyObject, &ModifyObject::nodeNameModification, this, &QSofaListView::nodeNameModification );
        connect ( dialogModifyObject, &ModifyObject::dataModified, this, &QSofaListView::dataModified );
        dialogModifyObject->show();
        dialogModifyObject->raise();
    }
}

void QSofaListView::UpdateOpenedDialogs()
{
    std::map<void*,QDialog*>::const_iterator iter;
    for(iter = map_modifyObjectWindow.begin(); iter != map_modifyObjectWindow.end() ; ++iter)
    {
        ModifyObject* modify = reinterpret_cast<ModifyObject*>(iter->second);
        modify->updateTables();
    }
}

void QSofaListView::ExpandRootNodeOnly()
{
    this->expandToDepth(0);
}

/// @brief Open a file at given path and line number using an external editor.
///
/// The external editor is defined in a QSettings with the following entries:
/// [General]
/// ExternalEditor=qtcreator
/// ExternalEditorParams=-client ${filename}:${fileno}
/// where ${filename} is expanded with the full path to the file
/// where ${fileno} is expanded with the line number to open at.
void openInExternalEditor(const std::string filename, const int fileloc)
{
    const QFileInfo f(filename.c_str());

    const std::string settingsFile = BaseGUI::getConfigDirectoryPath() + "/QSettings.ini";
    QSettings settings(settingsFile.c_str(), QSettings::IniFormat);

    /// In case the setting file does not contains the needed entries, let's put default ones
    /// based on qtcreator.
    if(!settings.contains("ExternalEditor"))
        settings.setValue("ExternalEditor", "qtcreator");
    if(!settings.contains("ExternalEditorParams"))
        settings.setValue("ExternalEditorParams", "-client ${filename}:${fileno}");

    const QString editor = settings.value("ExternalEditor").toString();
    QString params = settings.value("ExternalEditorParams").toString();

    params.replace("${filename}", f.absoluteFilePath());
    params.replace("${fileno}", QString::number(fileloc));
    const QStringList paramsAsList = params.split(QRegularExpression("(\\ )"));
    if ( QProcess::execute(editor, paramsAsList) != 0 )
    {
        msg_warning("QSofaListView") << "Unable to execute \"" << editor.toStdString() << " "
                                     << params.toStdString() << "\"" << msgendl
                                     << "  The file will NOT be opened at the right line." << msgendl
                                     << "  Set your preferred editor in: " << settingsFile << msgendl
                                     << "  Falling back to your system default editor.";

        QDesktopServices::openUrl(QUrl::fromLocalFile( f.absoluteFilePath() ));
    }
}

void QSofaListView::openInstanciation()
{
    if(object_.isBase())
    {
        openInExternalEditor(object_.asBase()->getInstanciationSourceFileName(),
                             object_.asBase()->getInstanciationSourceFilePos());
    }
}

void QSofaListView::openImplementation()
{
    if(object_.isBase())
    {
        openInExternalEditor(object_.asBase()->getDefinitionSourceFileName(),
                             object_.asBase()->getDefinitionSourceFilePos());
    }
}


void QSofaListView::openInEditor()
{
    const QFileInfo finfo(QApplication::activeWindow()->windowFilePath());
    QDesktopServices::openUrl(QUrl::fromLocalFile(finfo.absoluteFilePath()));
}

void QSofaListView::copyFilePathToClipBoard()
{
    const QFileInfo finfo(QApplication::activeWindow()->windowFilePath());
    QApplication::clipboard()->setText(finfo.absoluteFilePath()) ;
}

/*****************************************************************************************************************/
// Test if a node can be erased in the graph : the condition is that none of its children has a menu modify opened
bool QSofaListView::isNodeErasable ( BaseNode* node)
{
    const QTreeWidgetItem* item = graphListener_->items[node];
    if(item == nullptr)
    {
        return false;
    }
    // check if there is already a dialog opened for that item in the graph
    std::map< void*, QTreeWidgetItem*>::iterator it;
    for (it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); ++it)
    {
        if (it->second == item) return false;
    }

    //check the item childs
    for(int i=0 ; i<item->childCount() ; i++)
    {
        const QTreeWidgetItem *child = item->child(i);
        for( it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); ++it)
        {
            if( it->second == child) return false;
        }
    }

    return true;

}

void QSofaListView::Export()
{
    Node* root = down_cast<Node>( graphListener_->findObject(this->topLevelItem(0))->toBaseNode() );
    GenGraphForm* form = new sofa::gui::qt::GenGraphForm(this);
    form->setScene ( root );
    std::string gname(((RealGUI*) (QApplication::topLevelWidgets()[0]))->windowFilePath().toStdString());
    const std::size_t gpath = gname.find_last_of("/\\");
    const std::size_t gext = gname.rfind('.');
    if (gext != std::string::npos && (gpath == std::string::npos || gext > gpath))
        gname = gname.substr(0,gext);
    form->filename->setText(gname.c_str());
    form->show();
}

} //namespace sofa::gui::qt
