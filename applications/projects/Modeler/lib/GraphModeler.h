/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GRAPHMODELER_H
#define SOFA_GRAPHMODELER_H

#include <deque>

#include "AddPreset.h"
#include "GraphHistoryManager.h"
#include "GlobalModification.h"
#include <sofa/core/SofaLibrary.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/gui/qt/GraphListenerQListView.h>


#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3TextDrag>
#include <Q3PopupMenu>
#include <Q3Header>
#else
#include <qheader.h>
#include <qlistview.h>
#include <qdragobject.h>
#include <qpopupmenu.h>
typedef QListViewItemIterator Q3ListViewItemIterator;
#endif

#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{


typedef sofa::core::ObjectFactory::ClassEntry ClassEntry;
typedef sofa::core::ObjectFactory::Creator    Creator;

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
typedef QTextDrag Q3TextDrag;
typedef QPopupMenu Q3PopupMenu;
#endif

using sofa::simulation::tree::GNode;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;
using namespace sofa::simulation::tree;
using sofa::core::SofaLibrary;

class GraphModeler : public Q3ListView
{
    friend class GraphHistoryManager;
    Q_OBJECT
public:
    GraphModeler( QWidget* parent=0, const char* name=0, Qt::WFlags f = 0 );
    ~GraphModeler();

    /// Set the Sofa Resources: intern library to get the creators of the elements
    void setSofaLibrary( SofaLibrary *l) { sofaLibrary = l;}

    /// Set a menu of Preset available when right clicking on a node
    void setPreset(Q3PopupMenu *_preset) {preset=_preset;}

    /// Return the Root of the simulation
    GNode *getRoot() {return graphRoot.get(); } //getGNode(firstChild());}

    /// Set the Root of the simulation
    GNode::SPtr setRoot(GNode::SPtr node=NULL, bool saveHistory=true) {clearGraph(saveHistory); return addGNode(NULL, node, saveHistory);}

    /// Clear the contents of the current Graph
    void clearGraph(bool saveHistory=true);

    /// Set the name of the simulation
    void setFilename(std::string filename) {filenameXML = filename;}
    std::string getFilename() {return filenameXML;}

    /// Keyboard Management
    void keyPressEvent ( QKeyEvent * e );

    template <class T>
    void getSelectedItems(T& selection)
    {
        Q3ListViewItemIterator it( this, Q3ListViewItemIterator::Selected );
        while ( it.current() )
        {
            //Verify if the parent item (node) is not selected
            Q3ListViewItem *currentItem = it.current();
            Q3ListViewItem *parentItem=currentItem->parent();
            bool parentNodeAlreadySelected=false;
            while (parentItem && !parentNodeAlreadySelected)
            {
                currentItem=parentItem;
                if (currentItem->isSelected())
                {
                    parentNodeAlreadySelected=true;
                    break;
                }
                parentItem=currentItem->parent();
            }
            if (!parentNodeAlreadySelected)
                selection.push_back(it.current());
            ++it;
        }
    }


    template <class T>
    void getComponentHierarchy(Q3ListViewItem *item, T &hierarchy)
    {
        if (!item) return;
        hierarchy.push_back(item);
        item = item->firstChild();
        if (!item) return;
        getComponentHierarchy(item, hierarchy);
        while (item->nextSibling())
        {
            item = item->nextSibling();
            getComponentHierarchy(item, hierarchy);
        }
    }

    /// Says if there is something to undo
    bool isUndoEnabled() {return historyManager->isUndoEnabled();}
    /// Says if there is something to redo
    bool isRedoEnabled() {return historyManager->isRedoEnabled();}

    /// Drag & Drop Management
    void dragEnterEvent( QDragEnterEvent* event);
    void dragMoveEvent( QDragMoveEvent* event);
    void dropEvent(QDropEvent* event);

    /// collapse all the nodes below the current one
    void collapseNode(Q3ListViewItem* item);
    /// expande all the nodes below the current one
    void expandNode(Q3ListViewItem* item);
    /// load a node as a child of the current one
    GNode::SPtr loadNode(Q3ListViewItem* item, std::string filename="", bool saveHistory=true);
    /// Save the whole graphe
    void save(const std::string &fileName);
    /// Save components
    void saveComponents(helper::vector<Q3ListViewItem*> items, const std::string &file);
    /// Open the window to configure a component
    void openModifyObject(Q3ListViewItem *);
    /// Delete a componnent
    void deleteComponent(Q3ListViewItem *item, bool saveHistory=true);
    /// Construct a node from a BaseElement, by passing the factory
    GNode::SPtr buildNodeFromBaseElement(GNode::SPtr node,xml::BaseElement *elem, bool saveHistory=false);
    void configureElement(Base* b, xml::BaseElement *elem);

    /// Used to know what component is about to be created by a drag&drop
    void setLastSelectedComponent( const std::string& templateName, ClassEntry* entry) {lastSelectedComponent = std::make_pair(templateName, entry);}


signals:
    void fileOpen(const QString&);

    void operationPerformed(GraphHistoryManager::Operation &);
    void undo();
    void redo();
    void graphClean();

    void undoEnabled(bool);
    void redoEnabled(bool);
    void graphModified(bool);
    void displayMessage(const std::string &);

public slots:
    void editUndo() {emit undo();}
    void editRedo() {emit redo();};

    bool editCut(std::string path);
    bool editCopy(std::string path);
    bool editPaste(std::string path);

    //Right Click Menu
#ifdef SOFA_QT4
    void doubleClick(Q3ListViewItem *);
    void rightClick(Q3ListViewItem *, const QPoint &, int );
#else
    void doubleClick(QListViewItem *);
    void rightClick(QListViewItem *, const QPoint &, int );
#endif
    /// Context Menu Operation: collasping all the nodes below the current one
    void collapseNode();
    /// Context Menu Operation: expanding all the nodes below the current one
    void expandNode();
    /// Context Menu Operation: loading a node as a child of the current one
    GNode::SPtr loadNode();
    /// Context Menu Operation: process to a global modification of a Data
    void globalModification();

    /// Load a file given the node in which it will be added
    GNode::SPtr loadNode(GNode::SPtr, std::string, bool saveHistory=true);
    /// Context Menu Operation: loading a preset: open the window of configuration
    void loadPreset(std::string presetName);
    /// Context Menu Operation: loading a preset: actually creating the node, given its parameters (path to files, and initial position)
    void loadPreset(GNode*,std::string,std::string*, std::string,std::string,std::string);
    /// Context Menu Operation: Saving the selection
    void saveComponents();
    /// Context Menu Operation: Open the window to configure a component
    void openModifyObject();
    /// Context Menu Operation: Deleting a componnent
    void deleteComponent();

    /// Close all opened configuration windows
    void closeDialogs();
    /// Unlock a component: the configuration window has just closed
    void modifyUnlock ( void *Id );

protected:

    bool getSaveFilename(std::string &filename);
    /// Given a position, get the GNode corresponding (if the point is on a component, it returns the GNode parent)
    GNode      *getGNode(const QPoint &pos) const;

    /// Given a item of the list, return the GNode corresponding
    Base      *getComponent(Q3ListViewItem *item) const;
    /// Given a item of the list, return the GNode corresponding
    GNode      *getGNode(Q3ListViewItem *item) const;
    /// Get the component corresponding to the item, NULL if the item is a GNode
    BaseObject *getObject(Q3ListViewItem *item) const;

    /// Insert a GNode in the scene
    GNode::SPtr addGNode(GNode::SPtr parent, GNode::SPtr node=NULL, bool saveHistory=true);
    /// Insert a Component in the scene
    BaseObject::SPtr addComponent(GNode::SPtr parent, const ClassEntry* entry, const std::string& templateName, bool saveHistory=true, bool displayWarning=true );

    void changeComponentDataValue(const std::string &name, const std::string &value, Base* component) const ;

    /// Find the Sofa Component above the item
    Base *getComponentAbove(Q3ListViewItem *item);
    /// Set a dropped component in the right position in the graph
    void initItem(Q3ListViewItem *item, Q3ListViewItem *above);
    /// Move an item (and the sofa component corresponding) above the other Q3ListViewItem "above"
    void moveItem(Q3ListViewItem *item, Q3ListViewItem *above);

    /// Verify if no component is being edited, starting from the current GNode passed, and going through all the children
    bool isNodeErasable ( core::objectmodel::BaseNode* element );
    /// Verigy if the present component is being edited
    bool isObjectErasable ( core::objectmodel::Base* element );
    /// Change a preset node, update the paths to the files and the initial position
    void updatePresetNode(xml::BaseElement &elem, std::string meshFile, std::string translation, std::string rotation, std::string scale);

    GraphListenerQListView *graphListener; // Management of the list: Listener of the sofa tree
    GNode::SPtr graphRoot; ///< root node of the graph (it is now necessary to hold a smart pointer to it in order to keep it from being deleted)
    SofaLibrary *sofaLibrary;
    Q3PopupMenu *preset;  //Preset menu selection appearing when right click on a node
    AddPreset *DialogAdd; //Single Window appearing when adding a preset

    //Modify windows management: avoid duplicity, and dependencies
    void *current_Id_modifyDialog;
    std::map< void*, Q3ListViewItem* >       map_modifyDialogOpened;
    std::map< void*, QDialog* >    map_modifyObjectWindow;

    std::string filenameXML; //name associated to the current graph

    //Store template + ClassEntry
    std::pair< std::string, ClassEntry* > lastSelectedComponent;

    GraphHistoryManager *historyManager;
};

}
}
}

#endif
