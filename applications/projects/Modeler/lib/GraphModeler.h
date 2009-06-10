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
#include <sofa/core/SofaLibrary.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/gui/qt/GraphListenerQListView.h>
#include <sofa/gui/qt/ModifyObject.h>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3TextDrag>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpopupmenu.h>
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
using namespace sofa::simulation::tree;
using sofa::core::SofaLibrary;

class GraphModeler : public Q3ListView
{

    Q_OBJECT
public:
    GraphModeler( QWidget* parent=0, const char* name=0, Qt::WFlags f = 0 ):Q3ListView(parent, name, f), graphListener(NULL)
    {
        graphListener = new GraphListenerQListView(this);
        addColumn("Graph");
        header()->hide();
        setSorting ( -1 );

#ifdef SOFA_QT4
        connect(this, SIGNAL(doubleClicked ( Q3ListViewItem *, const QPoint &, int )), this, SLOT( doubleClick(Q3ListViewItem *)));
        connect(this, SIGNAL(rightButtonClicked ( Q3ListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(Q3ListViewItem *, const QPoint &, int )));
#else
        connect(this, SIGNAL(doubleClicked ( QListViewItem *, const QPoint &, int )), this, SLOT( doubleClick(QListViewItem *)));
        connect(this, SIGNAL(rightButtonClicked ( QListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(QListViewItem *, const QPoint &, int )));
#endif
        DialogAdd=NULL;
    };

    ~GraphModeler()
    {
        for (unsigned int i=0; i<historyOperation.size(); ++i) editUndo();
        simulation::getSimulation()->unload(getRoot());

        delete graphListener;
        if (DialogAdd) delete DialogAdd;
    }

    /// Set the Sofa Resources: intern library to get the creators of the elements
    void setSofaLibrary( SofaLibrary *l) { sofaLibrary = l;}

    /// Set a menu of Preset available when right clicking on a node
    void setPreset(Q3PopupMenu *_preset) {preset=_preset;}

    /// Return the Root of the simulation
    GNode *getRoot() {return getGNode(firstChild());}

    /// Set the Root of the simulation
    GNode *setRoot(GNode *node=NULL, bool saveHistory=true) {clearGraph(saveHistory); return addGNode(NULL, node, saveHistory);}

    /// Clear the contents of the current Graph
    void clearGraph(bool saveHistory=true);

    /// Set the name of the simulation
    void setFilename(std::string filename) {filenameXML = filename;}
    std::string getFilename() {return filenameXML;}

    /// Keyboard Management
    void keyPressEvent ( QKeyEvent * e );

    /// Says if there is something to undo
    bool isUndoEnabled() {return  historyOperation.size();}
    /// Says if there is something to redo
    bool isRedoEnabled() {return historyUndoOperation.size();}

    /// Drag & Drop Management
    void dragEnterEvent( QDragEnterEvent* event);
    void dragMoveEvent( QDragMoveEvent* event);
    void dropEvent(QDropEvent* event);

    /// collapse all the nodes below the current one
    void collapseNode(Q3ListViewItem* item);
    /// expande all the nodes below the current one
    void expandNode(Q3ListViewItem* item);
    /// load a node as a child of the current one
    GNode *loadNode(Q3ListViewItem* item, std::string filename="");
    /// Save a node: call the GUI to get the file name
    void saveNode(Q3ListViewItem* item);
    /// Directly save a node
    void saveNode(GNode* node, std::string file);
    /// Save a component
    void saveComponent(BaseObject* object, std::string file);
    /// Open the window to configure a component
    void openModifyObject(Q3ListViewItem *);
    /// Delete a componnent
    void deleteComponent(Q3ListViewItem *item, bool saveHistory=true);
    /// Construct a node from a BaseElement, by passing the factory
    GNode *buildNodeFromBaseElement(GNode *node,xml::BaseElement *elem, bool saveHistory=false);
    void configureElement(Base* b, xml::BaseElement *elem);

    /// Used to know what component is about to be created by a drag&drop
    void setLastSelectedComponent( const std::string& templateName, ClassEntry *entry) {lastSelectedComponent = std::make_pair(templateName, entry);}

signals:
    void fileOpen(const QString&);
    void undo(bool);
    void redo(bool);

public slots:
    void editUndo();
    void editRedo();

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
    GNode *loadNode();
    /// Load a file given the node in which it will be added
    GNode *loadNode(GNode*, std::string);
    /// Context Menu Operation: loading a preset: open the window of configuration
    void loadPreset(std::string presetName);
    /// Context Menu Operation: loading a preset: actually creating the node, given its parameters (path to files, and initial position)
    void loadPreset(GNode*,std::string,std::string*, std::string*,std::string*,std::string);
    /// Context Menu Operation: Saving a node
    void saveNode();
    /// Context Menu Operation: Open the window to configure a component
    void openModifyObject();
    /// Context Menu Operation: Deleting a componnent
    void deleteComponent();

    /// Close all opened configuration windows
    void closeDialogs();
    /// Unlock a component: the configuration window has just closed
    void modifyUnlock ( void *Id );

protected:
    /// Given a position, get the GNode corresponding (if the point is on a component, it returns the GNode parent)
    GNode      *getGNode(const QPoint &pos);

    /// Given a item of the list, return the GNode corresponding
    GNode      *getGNode(Q3ListViewItem *item);
    /// Get the component corresponding to the item, NULL if the item is a GNode
    BaseObject *getObject(Q3ListViewItem *item);

    /// Insert a GNode in the scene
    GNode      *addGNode(GNode *parent, GNode *node=NULL, bool saveHistory=true);
    /// Insert a Component in the scene
    BaseObject *addComponent(GNode *parent, const ClassEntry *entry, const std::string& templateName, bool saveHistory=true, bool displayWarning=true );

    /// Find the Sofa Component above the item
    Base *getComponentAbove(Q3ListViewItem *item);
    /// Set a dropped component in the right position in the graph
    void initItem(Q3ListViewItem *item, Q3ListViewItem *above);
    /// Move an item (and the sofa component corresponding) above the other Q3ListViewItem "above"
    void moveItem(Q3ListViewItem *item, Q3ListViewItem *above);

    /// Verify if no component is being edited, starting from the current GNode passed, and going through all the children
    bool isNodeErasable ( core::objectmodel::Base* element );
    /// Verigy if the present component is being edited
    bool isObjectErasable ( core::objectmodel::Base* element );
    /// Change a preset node, update the paths to the files and the initial position
    void updatePresetNode(xml::BaseElement &elem, std::string meshFile, std::string *translation, std::string *rotation, std::string scale);

    GraphListenerQListView *graphListener; // Management of the list: Listener of the sofa tree
    SofaLibrary *sofaLibrary;
    Q3PopupMenu *preset;  //Preset menu selection appearing when right click on a node
    AddPreset *DialogAdd; //Single Window appearing when adding a preset

    //Modify windows management: avoid duplicity, and dependencies
    void *current_Id_modifyDialog;
    std::map< void*, Base* >       map_modifyDialogOpened;
    std::map< void*, QDialog* >    map_modifyObjectWindow;

    std::string filenameXML; //name associated to the current graph

    //Store template + ClassEntry
    std::pair< std::string, ClassEntry* > lastSelectedComponent;

    //-----------------------------------------------------------------------------//
    //Historic of actions: management of the undo/redo actions
    ///Basic class storing information about the operation done
    class Operation
    {
    public:
        Operation() {};
        enum op {DELETE_OBJECT,DELETE_GNODE, ADD_OBJECT,ADD_GNODE};
        Operation(Base* sofaComponent_,  op ID_): sofaComponent(sofaComponent_), above(NULL), ID(ID_)
        {}

        Base* sofaComponent;
        GNode* parent;
        Base* above;
        op ID;
        std::string info;
    };


    void storeHistory(Operation &o);
    void processUndo(Operation &o);

    void clearHistory();
    void clearHistoryUndo();

    std::vector< Operation > historyOperation;
    std::vector< Operation > historyUndoOperation;
    //-----------------------------------------------------------------------------//

};













///Overloading ModifyObject to display all the elements
class ModifyObjectModeler: public ModifyObject
{
public:
    ModifyObjectModeler( void *Id_, core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked, QWidget* parent_, const char* name= 0 )
    {
        parent = parent_;
        node = NULL;
        Id = Id_;
        visualContentModified=false;
        setCaption(name);
        HIDE_FLAG = false;
        READONLY_FLAG=false; //everything will be editable
        EMPTY_FLAG = true;
        RESIZABLE_FLAG = true;
        REINIT_FLAG = false;
        //remove the qwt graphes
        energy_curve[0]=energy_curve[1]=energy_curve[2]=NULL;
        outputTab = warningTab = NULL;
        logWarningEdit=NULL; logOutputEdit=NULL;
        graphEnergy=NULL;
        //Initialization of the Widget
        setNode(node_clicked, item_clicked);
        connect ( this, SIGNAL( dialogClosed(void *) ) , parent_, SLOT( modifyUnlock(void *)));
    }

};

}
}
}

#endif
