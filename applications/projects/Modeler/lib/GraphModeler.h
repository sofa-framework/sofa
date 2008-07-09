/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3TextDrag>
#else
#include <qlistview.h>
#include <qdragobject.h>
#endif
#include <sofa/gui/qt/GraphListenerQListView.h>
#include <sofa/gui/qt/ModifyObject.h>
#include <sofa/simulation/tree/xml/NodeElement.h>
#include <sofa/simulation/tree/xml/ObjectElement.h>

#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
typedef QTextDrag Q3TextDrag;
#endif

typedef sofa::core::ObjectFactory::ClassEntry ClassInfo;
typedef sofa::core::ObjectFactory::Creator    ClassCreator;
using sofa::simulation::tree::GNode;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation::tree;

class GraphModeler : public Q3ListView
{

    typedef std::map< const QObject* , std::pair< ClassInfo*, QObject*> > ComponentMap;

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
    };

    void clearGraph();
    void setFilename(std::string filename) {filenameXML = filename;}
    std::string getFilename() {return filenameXML;}

    void dragEnterEvent( QDragEnterEvent* event)
    {
        QString text;
        Q3TextDrag::decode(event, text);
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file") {event->accept();}
        else
        {
            if ( getGNode(event->pos()))
                event->accept(event->answerRect());
            else
                event->ignore(event->answerRect());
        }
    }
    void dragMoveEvent( QDragMoveEvent* event)
    {
        QString text;
        Q3TextDrag::decode(event, text);
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file") {event->accept();}
        else
        {
            if ( getGNode(event->pos()))
                event->accept(event->answerRect());
            else
                event->ignore(event->answerRect());
        }
    }

    void dropEvent(QDropEvent* event);

    void setLibrary(ComponentMap &s) {library=s;}


    GNode *getGNode(const QPoint &pos);
    GNode *getGNode(Q3ListViewItem *item);
    GNode *getRoot() {return getGNode(firstChild());}
    BaseObject *getObject(Q3ListViewItem *item);

    void keyPressEvent ( QKeyEvent * e );

signals:
    void fileOpen(std::string);


public slots:
    void collapseNode();
    void collapseNode(Q3ListViewItem* item);
    void expandNode();
    void expandNode(Q3ListViewItem* item);
    void loadNode();
    void loadNode(Q3ListViewItem* item);
    void saveNode();
    void saveNode(Q3ListViewItem* item);
    void openModifyObject();
    void openModifyObject(Q3ListViewItem *);
#ifdef SOFA_QT4
    void doubleClick(Q3ListViewItem *);
    void rightClick(Q3ListViewItem *, const QPoint &, int );
#else
    void doubleClick(QListViewItem *);
    void rightClick(QListViewItem *, const QPoint &, int );
#endif
    GNode *addGNode(GNode *parent, GNode *node=NULL, bool saveHistory=true);
    BaseObject *addComponent(GNode *parent, ClassInfo *entry, std::string templateName, bool saveHistory=true );
    void deleteComponent();
    void deleteComponent(Q3ListViewItem *item, bool saveHistory=true);
    void modifyUnlock ( void *Id );


    void editUndo();
    void editRedo();
protected:

    bool isNodeErasable ( core::objectmodel::Base* element );
    bool isObjectErasable ( core::objectmodel::Base* element );

    class Operation
    {
    public:
        Operation() {};
        enum op {DELETE_OBJECT, ADD_OBJECT};
        Operation(Q3ListViewItem* item_,Base* sofaComponent_,  op ID_):
            item(item_),sofaComponent(sofaComponent_), ID(ID_)
        {}
        Q3ListViewItem* item;
        Base* sofaComponent;
        op ID;
    };


    GraphListenerQListView *graphListener;
    ComponentMap library;


    //Modify windows management: avoid duplicity, and dependencies
    void *current_Id_modifyDialog;
    std::map< void*, Base* >       map_modifyDialogOpened;
    std::map< void*, QDialog* >    map_modifyObjectWindow;

    std::string filenameXML; //name associated to the current graph
    std::deque< Operation > historyOperation;
    std::deque< Operation >::iterator currentStateHistory;

};













//Overloading ModifyObject to display all the elements
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
        EMPTY_FLAG = true;
        RESIZABLE_FLAG = true;
        REINIT_FLAG = false;

        energy_curve[0]=NULL;	        energy_curve[1]=NULL;	        energy_curve[2]=NULL;
        //Initialization of the Widget
        setNode(node_clicked, item_clicked);
        connect ( this, SIGNAL( dialogClosed(void *) ) , parent_, SLOT( modifyUnlock(void *)));
    }

};

}
}
}

#endif
