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

#include <sofa/gui/SofaGUI.h>
#include <sofa/helper/system/glut.h>
#include <sofa/gui/qt/RealGUI.h>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3TextDrag>
#include <QPushButton>
#include <QTabWidget>
#include <QTabBar>
#include <Q3PopupMenu>
#else
#include <qlistview.h>
#include <qdragobject.h>
#include <qpushbutton.h>
#include <qtabwidget.h>
#include <qtabbar.h>
#include <qpopupmenu.h>
#endif


namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QPopupMenu Q3PopupMenu;
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

signals:
    void loadPresetGraph(std::string);


public:

    /// Create a new empty Tab
    void createTab();


public slots:
    void test() {std::cout << "!!!!!!!!!!!!!!\n";}
    void dragComponent();
    void changeComponent(ClassInfo *currentComponent);
#ifdef SOFA_QT4
    void changeInformation(Q3ListViewItem *);
#else
    void changeInformation(QListViewItem *);
#endif

    /// Dropping a Node in the Graph
    void newGNode();

    /// Creation of a new scene (new tab will be created)
    void fileNew() {fileNew(NULL);};
    void fileNew(GNode* root);

    /// Open an existing simulation (new tab will be created)
    void fileOpen();
    void fileOpen(std::string filename);
    void fileOpen(const QString &filename) {fileOpen(std::string(filename.ascii()));}

    /// Save the current simulation
    void fileSave();
    void fileSave(std::string filename);
    void fileSaveAs();

    /// Remove all components of the current simulation
    void clearTab();
    /// Close the current simulation
    void closeTab();
    /// Create a new tab containing an empty simulation (by default the collision pipeline is added)
    void newTab();

    /// Quit the Modeler
    void fileExit() {close();};

    /// Launch the current simulation into Sofa
    void runInSofa();

    /// Change of simulation by changing the current opened tabulation
    void changeCurrentScene( QWidget*);
    /// Change the name of the main window
    void changeNameWindow(std::string filename);

    /// not implemented yet: would undo the last operation done in the current simulation
    void editUndo() {graph->editUndo();}
    /// not implemented yet: would redo the last operation done in the current simulation
    void editRedo() {graph->editRedo();}

    /// Load a preset stored in the menu preset: add a node to the current simulation
    void loadPreset(int);

    /// From the name of the type of a component, gives serveral information
    ClassInfo* getInfoFromName(std::string name);


    void dragEnterEvent( QDragEnterEvent* event)
    {
        QString text;
        Q3TextDrag::decode(event, text);
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file")  event->accept();
        else          	 event->ignore();
    }

    void dragMoveEvent( QDragMoveEvent* event)
    {

        QString text;
        Q3TextDrag::decode(event, text);
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file")  event->accept();
        else          	 event->ignore();
    }

    void dropEvent(QDropEvent* event);
    void keyPressEvent ( QKeyEvent * e );

    /// Open a recently Opened files from the menu Recently Opened Files...
    void fileRecentlyOpened(int id);
    /// Update the menu Recently Opened Files...
    void updateRecentlyOpened(std::string fileLoaded);

protected:
    /// Widget containing all the graphs
    QTabWidget *sceneTab;
    /// Current in-use graph
    GraphModeler *graph; //currentGraph in Use
    /// Current opened Tab
    QWidget *tabGraph;
    /// Menu recently opened
    //Q3PopupMenu *recentlyOpened;
    /// Menu preset
    Q3PopupMenu *preset;
    /// Correspondance between a name clicked in the menu and a path to the preset
    std::map< std::string, std::string > mapPreset;


    std::map< const QObject* , std::pair<ClassInfo*, QObject*> > mapComponents;
    std::map< const QWidget*, GraphModeler*> mapGraph;
    std::map< const QWidget*, sofa::gui::qt::RealGUI*> mapSofa;

private:
    std::string presetPath;
    std::string examplePath;
};
}
}
}
#endif
