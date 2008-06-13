#include "GraphModeler.h"

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/gui/qt/FileManagement.h> //static functions to manage opening/ saving of files
#ifdef SOFA_QT4
#include <Q3Header>
#include <Q3PopupMenu>
#else
#include <qheader.h>
#include <qpopupmenu.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

using namespace sofa::simulation::tree;

#ifdef SOFA_QT4
typedef Q3PopupMenu QPopupMenu;
#endif
void GraphModeler::addGNode(GNode *parent, bool saveHistory)
{
    GNode *child = new GNode();
    parent->addChild(child);
    graphListener->addChild(parent, child);
    if (saveHistory)
    {
        Operation adding(graphListener->items[child], child, Operation::ADD_OBJECT);
        historyOperation.push_front(adding);
    }
}

void GraphModeler::addComponent(GNode *parent, ClassInfo* entry, std::string templateName, bool saveHistory)
{
    if (!parent || !entry) return;
    BaseObject *object;
    if (entry->creatorMap.find(entry->defaultTemplate) != entry->creatorMap.end())
    {
        if (templateName.empty())
            object = entry->creatorMap.find(entry->defaultTemplate)->second->createInstance(NULL, NULL);
        else
            object = entry->creatorMap.find(templateName)->second->createInstance(NULL, NULL);

    }
    else
        object = entry->creatorList.begin()->second->createInstance(NULL, NULL);

    parent->addObject(object);
    graphListener->addObject(parent, object);

    if (saveHistory)
    {
        Operation adding(graphListener->items[object], object, Operation::ADD_OBJECT);
        historyOperation.push_front(adding);
    }
}

void GraphModeler::dropEvent(QDropEvent* event)
{

    QString text;
    QTextDrag::decode(event, text);
    if (library.find(event->source()) != library.end())
    {
        std::string templateName =  text.ascii();
        addComponent(getGNode(event->pos()), library.find(event->source())->second.first, templateName );
    }
    else
    {
        if (text == QString("GNode"))
        {
            GNode* node=getGNode(event->pos());
            if (node)  addGNode(node);
        }
    }
}





BaseObject *GraphModeler::getObject(QListViewItem *item)
{
    std::map<core::objectmodel::Base*, QListViewItem* >::iterator it;
    for (it = graphListener->items.begin(); it != graphListener->items.end(); it++)
    {
        if (it->second == item)
        {
            return dynamic_cast< BaseObject *>(it->first);
        }
    }
    return NULL;
}


GNode *GraphModeler::getGNode(const QPoint &pos)
{
    QListViewItem *item = itemAt(pos);
    if (!item) return NULL;
    return getGNode(item);
}



GNode *GraphModeler::getGNode(QListViewItem *item)
{
    if (!item) return NULL;
    sofa::core::objectmodel::Base *object;
    std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it;
    for (it = graphListener->items.begin(); it != graphListener->items.end(); it++)
    {
        if (it->second == item)
        {
            object = it->first;
            break;
        }
    }
    if (it == graphListener->items.end()) return NULL;

    if (dynamic_cast<GNode*>(it->first)) return dynamic_cast<GNode*>(it->first);
    else
    {
        item = item->parent();
        for (it = graphListener->items.begin(); it != graphListener->items.end(); it++)
        {
            if (it->second == item)
            {
                object = it->first;
                break;
            }
        }
        if (it == graphListener->items.end()) return NULL;
        if (dynamic_cast<GNode*>(it->first)) return dynamic_cast<GNode*>(it->first);
        else return NULL;
    }
}


void GraphModeler::openModifyObject()
{
    QListViewItem *item = currentItem();
    openModifyObject(item);
}

void GraphModeler::openModifyObject(QListViewItem *item)
{
    if (!item) return;

    std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it;
    for (it = graphListener->items.begin(); it != graphListener->items.end(); it++)
    {
        if (it->second == item)
        {
            break;
        }
    }
    if (it == graphListener->items.end()) return;


    //Unicity and identification of the windows
    current_Id_modifyDialog = it->first;
    std::map< void*, QDialog* >::iterator testWindow =  map_modifyObjectWindow.find( current_Id_modifyDialog);
    if ( testWindow != map_modifyObjectWindow.end())
    {
        //Object already being modified: no need to open a new window
        (*testWindow).second->raise();
        return;
    }

    ModifyObjectModeler *dialogModify = new ModifyObjectModeler ( current_Id_modifyDialog, it->first, item,this,item->text(0));
    map_modifyObjectWindow.insert( std::make_pair(current_Id_modifyDialog, dialogModify));
    //If the item clicked is a node, we add it to the list of the element modified
    if ( dynamic_cast<GNode *> (it->first ) )
        map_modifyDialogOpened.insert ( std::make_pair ( current_Id_modifyDialog, it->first ) );
    else
        map_modifyDialogOpened.insert ( std::make_pair ( current_Id_modifyDialog, getGNode(item) ) );

    dialogModify->show();
    dialogModify->raise();
}

void GraphModeler::doubleClick(QListViewItem *item)
{
    if (!item) return;
    item->setOpen ( !item->isOpen() );
    openModifyObject(item);

}
void GraphModeler::rightClick(QListViewItem *item, const QPoint &point, int index)
{
    if (!item) return;


    QPopupMenu *contextMenu = new QPopupMenu ( this, "ContextMenu" );
    if (item->childCount() != 0)
    {
        contextMenu->insertItem("Collapse", this, SLOT( collapseNode()));
        contextMenu->insertItem("Expand"  , this, SLOT( expandNode()));
        contextMenu->insertSeparator ();
        contextMenu->insertItem("Save"  , this, SLOT( saveNode()));
    }
    contextMenu->insertItem("Delete"  , this, SLOT( deleteComponent()));
    contextMenu->insertItem("Modify"  , this, SLOT( openModifyObject()));
    contextMenu->popup ( point, index );

}


void GraphModeler::collapseNode()
{
    collapseNode(currentItem());
}

void GraphModeler::collapseNode(QListViewItem* item)
{
    if (!item) return;

    QListViewItem* child;
    child = item->firstChild();
    while ( child != NULL )
    {
        child->setOpen ( false );
        child = child->nextSibling();
    }
    item->setOpen ( true );
}

void GraphModeler::expandNode()
{
    expandNode(currentItem());
}

void GraphModeler::expandNode(QListViewItem* item)
{
    if (!item) return;

    item->setOpen ( true );
    if ( item != NULL )
    {
        QListViewItem* child;
        child = item->firstChild();
        while ( child != NULL )
        {
            item = child;
            child->setOpen ( true );
            expandNode(item);
            child = child->nextSibling();
        }
    }
}

void GraphModeler::saveNode()
{
    saveNode(currentItem());
}

void GraphModeler::saveNode(QListViewItem* item)
{
    if (!item) return;
    GNode *node = getGNode(item);
    if (!node) return;

    QString s = sofa::gui::qt::getSaveFileName ( this, NULL, "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
        getSimulation()->printXML(node, s.ascii());
}

void GraphModeler::deleteComponent(QListViewItem* item, bool saveHistory)
{
    if (!item) return;

    GNode *parent = getGNode(item->parent());
    if (item->childCount() == 0 && item != graphListener->items[getGNode(item)])
    {
        BaseObject* object = getObject(item);
        graphListener->removeObject(getGNode(item), getObject(item));

        if (saveHistory)
        {
            Operation removal(item, getObject(item),Operation::DELETE_OBJECT);
            historyOperation.push_front(removal);
        }
        //OPERATION DELETE: to remove in order to be able to undo operation
        parent->removeObject(object);
        delete object;
    }
    else
    {

        GNode *node = getGNode(item);
        graphListener->removeChild(parent, node);

        if (saveHistory)
        {
            Operation removal(item, getGNode(item),Operation::DELETE_OBJECT);
            historyOperation.push_front(removal);
        }

        //OPERATION DELETE: to remove in order to be able to undo operation
        getSimulation()->unload (node);


        //if we have removed the root, we recreate a new one
        if (!parent) fileNew();
    }

}
void GraphModeler::deleteComponent()
{
    QListViewItem *item = currentItem();
    deleteComponent(item);
}

void GraphModeler::modifyUnlock ( void *Id )
{
    map_modifyDialogOpened.erase( Id );
    map_modifyObjectWindow.erase( Id );
}


void GraphModeler::changeName(std::string filename)
{
    filenameXML = filename;
    emit( changeNameWindow(filename) );
}

void GraphModeler::fileNew(GNode* root)
{

    if (!root) filenameXML.clear();
    changeName(filenameXML);
    GNode *current_root=getGNode(firstChild());
    if (current_root)
    {
        graphListener->removeChild(NULL,current_root);
        getSimulation()->unload(current_root);
    }
    if (!root) { root = new GNode(); root->setName("Root");}
    graphListener->addChild(NULL, root);
    firstChild()->setExpandable(true);
    firstChild()->setOpen(true);
    historyOperation.clear();
    currentStateHistory=historyOperation.end();
}


void GraphModeler::fileReload()
{
    fileOpen(filenameXML);
}
void GraphModeler::fileOpen()
{
    QString s = getOpenFileName ( this, NULL,"Scenes (*.scn *.xml *.simu *.pscn)", "open file dialog",  "Choose a file to open" );
    if (s.length() >0)
        fileOpen(s.ascii());
}
void GraphModeler::fileOpen(std::string filename)
{
    filenameXML = filename;
    GNode *root = NULL;
    if (!filenameXML.empty())
        root = getSimulation()->load ( filename.c_str() );
    fileNew(root);
}

void GraphModeler::fileSave()
{
    if (filenameXML.empty()) fileSaveAs();
    else 	                   fileSave(filenameXML);
}

void GraphModeler::fileSave(std::string filename)
{
    changeName(filename);
    getSimulation()->printXML(getGNode(firstChild()), filename.c_str());
}

void GraphModeler::fileSaveAs()
{
    QString s = sofa::gui::qt::getSaveFileName ( this, NULL, "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
        fileSave ( s.ascii() );
}

void GraphModeler::editUndo()
{
    std::cout << "Undo \n";
    std::cout << historyOperation.size() << " size of the history\n";
    if (currentStateHistory == historyOperation.begin()) return;
    currentStateHistory--;
}

void GraphModeler::editRedo()
{
    std::cout << "Redo \n";
}

void GraphModeler::keyPressEvent ( QKeyEvent * e )
{
    switch ( e->key() )
    {
    case Qt::Key_Delete :
    {
        deleteComponent();
        break;
    }
    default:
    {
        e->ignore();
        break;
    }
    }
}

}
}
}
