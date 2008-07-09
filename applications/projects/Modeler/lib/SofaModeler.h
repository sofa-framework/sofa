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

#ifndef SOFA_MODELER_H
#define SOFA_MODELER_H


#include "Modeler.h"
#include "GraphModeler.h"
#include <map>
#include <vector>
#include <string>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/tree/GNode.h>


#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3TextDrag>
#include <QPushButton>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpushbutton.h>
#endif


namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QListView Q3ListView;
#endif

typedef sofa::core::ObjectFactory::ClassEntry ClassInfo;
typedef sofa::core::ObjectFactory::Creator    ClassCreator;

using sofa::simulation::tree::GNode;


class SofaModeler : public ::Modeler
{

    Q_OBJECT
public :

    SofaModeler();
    ~SofaModeler() {};

public slots:
    void dragComponent();
    void changeComponent(ClassInfo *currentComponent);
#ifdef SOFA_QT4
    void changeInformation(Q3ListViewItem *);
#else
    void changeInformation(QListViewItem *);
#endif
    void newGNode();

    void fileNew() {fileNew(NULL);};
    void fileNew(GNode* root);

    void fileReload() {fileOpen(graph->getFilename());}

    void fileOpen();
    void fileOpen(std::string filename);

    void fileSave();
    void fileSave(std::string filename);
    void fileSaveAs();

    void fileExit() {exit(0);};

    void runInSofa();

    void changeNameWindow(std::string filename);
    void editUndo() {graph->editUndo();}
    void editRedo() {graph->editRedo();}
    ClassInfo* getInfoFromName(std::string name);


    void dragEnterEvent( QDragEnterEvent* event) {event->accept();}
    void dropEvent(QDropEvent* event);
    void keyPressEvent ( QKeyEvent * e );

    void fileRecentlyOpened(int id);
    void updateRecentlyOpened(std::string fileLoaded);

protected:
    //	  std::map< unsigned int, GraphModeler*> tabGraph;
    GraphModeler *graph; //currentGraph in Use

    std::map< const QObject* , std::pair<ClassInfo*, QObject*> > mapComponents;

};
}
}
}
#endif
