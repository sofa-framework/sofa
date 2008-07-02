#include "GraphModeler.h"

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/gui/qt/FileManagement.h> //static functions to manage opening/ saving of files
#include <sofa/helper/system/SetDirectory.h>
#ifdef SOFA_QT4
#include <Q3Header>
#include <Q3PopupMenu>
#include <QMessageBox>
#else
#include <qheader.h>
#include <qpopupmenu.h>
#include <qmessagebox.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{


#ifndef SOFA_QT4
typedef QPopupMenu Q3PopupMenu;
#endif
GNode *GraphModeler::addGNode(GNode *parent, bool saveHistory)
{
    GNode *child = new GNode();
    if (parent != NULL)
        parent->addChild(child);
    else
        child->setName("Root");

    graphListener->addChild(parent, child);

    if (saveHistory)
    {
        Operation adding(graphListener->items[child], child, Operation::ADD_OBJECT);
        historyOperation.push_front(adding);
    }
    return child;
}

BaseObject *GraphModeler::addComponent(GNode *parent, ClassInfo* entry, std::string templateName, bool saveHistory)
{
    BaseObject *object=NULL;;
    if (!parent || !entry) return object;



    xml::ObjectElement description("Default", entry->className.c_str() );

    if (!templateName.empty()) description.setAttribute("template", templateName.c_str());


    ClassCreator* c;

    if (entry->creatorMap.size() <= 1)
        c=entry->creatorMap.begin()->second;
    else
    {
        if (templateName.empty())
        {
            c=entry->creatorMap.find(entry->defaultTemplate)->second; templateName=entry->defaultTemplate;
        }
        else
            c=entry->creatorMap.find(templateName)->second;
    }

    if (c->canCreate(parent->getContext(), &description))
    {
        object = c->createInstance(parent->getContext(), NULL);
        graphListener->addObject(parent, object);
        if (saveHistory)
        {
            Operation adding(graphListener->items[object], object, Operation::ADD_OBJECT);
            historyOperation.push_front(adding);
        }
    }
    else
    {
        BaseObject* reference = parent->getContext()->getMechanicalState();

        if (!reference)
        {
            const QString caption("Creation Impossible");
            const QString warning=QString("No MechanicalState found in your Node ") + QString(parent->getName().c_str());
            if ( QMessageBox::warning ( this, caption,warning, QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape, QMessageBox::Ignore ) == QMessageBox::Cancel )
                return object;
        }
        else if (entry->className.find("Mapping") != std::string::npos) ;//we accept the mappings as no initialization of the object has been done
        else
        {
            const QString caption("Creation Impossible");
            const QString warning=
                QString("Your component won't be created: \n \t * <")
                + QString(reference->getTemplateName().c_str()) + QString("> DOFs are used in the Node ") + QString(parent->getName().c_str()) + QString("\n\t * <")
                + QString(templateName.c_str()) + QString("> is the type of your ") + QString(entry->className.c_str());
            if ( QMessageBox::warning ( this, caption,warning, QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape, QMessageBox::Ignore ) == QMessageBox::Cancel )
                return object;
        }
        object = c->createInstance(parent->getContext(), NULL);
        graphListener->addObject(parent, object);
    }
    return object;
}




void GraphModeler::dropEvent(QDropEvent* event)
{
    QString text;
    Q3TextDrag::decode(event, text);

    std::string filename(text.ascii());
    std::string test = filename; test.resize(4);
    if (test == "file")
    {

#ifdef WIN32
        for (unsigned int i=0; i<filename.size(); ++i)
        {
            if (filename[i] == '\\') filename[i] = '/';
        }
        filename = filename.substr(8); //removing file:///
#else
        filename = filename.substr(7); //removing file://
#endif
        filename.resize(filename.size()-1);
        filename[filename.size()-1]='\0';
        fileOpen(filename);
    }
    else
    {

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
}




BaseObject *GraphModeler::getObject(Q3ListViewItem *item)
{
    std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it;
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
    Q3ListViewItem *item = itemAt(pos);
    if (!item) return NULL;
    return getGNode(item);
}



GNode *GraphModeler::getGNode(Q3ListViewItem *item)
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
    Q3ListViewItem *item = currentItem();
    openModifyObject(item);
}

void GraphModeler::openModifyObject(Q3ListViewItem *item)
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

    map_modifyDialogOpened.insert ( std::make_pair ( current_Id_modifyDialog, it->first ) );

    dialogModify->show();
    dialogModify->raise();
}

void GraphModeler::doubleClick(Q3ListViewItem *item)
{
    if (!item) return;
    item->setOpen ( !item->isOpen() );
    openModifyObject(item);

}
void GraphModeler::rightClick(Q3ListViewItem *item, const QPoint &point, int index)
{
    if (!item) return;
    bool isNode=getObject(item)==NULL;

    Q3PopupMenu *contextMenu = new Q3PopupMenu ( this, "ContextMenu" );
    if (isNode)
    {
        contextMenu->insertItem("Collapse", this, SLOT( collapseNode()));
        contextMenu->insertItem("Expand"  , this, SLOT( expandNode()));
        contextMenu->insertSeparator ();
        contextMenu->insertItem("Save"  , this, SLOT( saveNode()));
    }
    int index_menu = contextMenu->insertItem("Delete"  , this, SLOT( deleteComponent()));

    if (isNode)
    {
        if ( !isNodeErasable ( getGNode(item) ) )
            contextMenu->setItemEnabled ( index_menu,false );
    }
    else
    {
        if ( !isObjectErasable ( getObject(item) ))
            contextMenu->setItemEnabled ( index_menu,false );
    }

    contextMenu->insertItem("Modify"  , this, SLOT( openModifyObject()));
    contextMenu->popup ( point, index );

}


void GraphModeler::collapseNode()
{
    collapseNode(currentItem());
}

void GraphModeler::collapseNode(Q3ListViewItem* item)
{
    if (!item) return;

    Q3ListViewItem* child;
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

void GraphModeler::expandNode(Q3ListViewItem* item)
{
    if (!item) return;

    item->setOpen ( true );
    if ( item != NULL )
    {
        Q3ListViewItem* child;
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

void GraphModeler::saveNode(Q3ListViewItem* item)
{
    if (!item) return;
    GNode *node = getGNode(item);
    if (!node) return;

    QString s = sofa::gui::qt::getSaveFileName ( this, NULL, "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
        getSimulation()->printXML(node, s.ascii());
}

void GraphModeler::deleteComponent(Q3ListViewItem* item, bool saveHistory)
{
    if (!item) return;

    GNode *parent = getGNode(item->parent());
    bool isNode   = getObject(item)==NULL;
    if (!isNode && isObjectErasable(getObject(item)))
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
        if (!isNodeErasable(getGNode(item))) return;
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
    Q3ListViewItem *item = currentItem();
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
    emit( updateRecentlyOpened(filename) );
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

    if (!root) { root = addGNode(NULL);}
    else graphListener->addChild(NULL, root);

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
    xml::BaseElement* newXML=NULL;
    if (!filenameXML.empty())
    {
        sofa::helper::system::SetDirectory chdir ( filename );
        newXML = xml::loadFromFile ( filename.c_str() );
        if (newXML == NULL) return;
        if (!newXML->init()) std::cerr<< "Objects initialization failed.\n";
        root = dynamic_cast<GNode*> ( newXML->getObject() );
    }
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
/*****************************************************************************************************************/
// Test if a node can be erased in the graph : the condition is that none of its children has a menu modify opened
bool GraphModeler::isNodeErasable ( core::objectmodel::Base* element )
{
    std::map< void*, core::objectmodel::Base*>::iterator it;
    for (it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); it++)
    {

        if (dynamic_cast< BaseObject* >(it->second))
        {
            if (getGNode(graphListener->items[it->second]) == element) return false;
        }
        else if (it->second == element) return false;

    }

    std::map< core::objectmodel::Base*, QListViewItem*>::iterator it_item;
    it_item = graphListener->items.find(element);

    QListViewItem *child = it_item->second->firstChild();
    while (child != NULL)
    {
        for (it_item = graphListener->items.begin(); it_item != graphListener->items.end(); it_item++)
        {
            if  (it_item->second == child)
            {
                if (!isNodeErasable(it_item->first)) return false;
                break;
            }
        }
        child = child->nextSibling();
    }
    return true;
}
bool GraphModeler::isObjectErasable ( core::objectmodel::Base* element )
{
    std::map< void*, core::objectmodel::Base*>::iterator it;
    for (it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); it++)
    {
        if (it->second == element) return false;
    }

    return true;
}

}
}
}
