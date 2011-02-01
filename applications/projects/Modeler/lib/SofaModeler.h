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

#ifndef SOFA_MODELER_H
#define SOFA_MODELER_H


#include "Modeler.h"
#include "GraphModeler.h"
#include "FilterLibrary.h"
#include "SofaTutorialManager.h"
#include <sofa/gui/qt/QMenuFilesRecentlyOpened.h>

#include <sofa/helper/Factory.h>

#ifdef SOFA_QT4
#include "QSofaTreeLibrary.h"
#include <Q3ListView>
#include <Q3TextDrag>
#include <QPushButton>
#include <QTabWidget>
#include <QTabBar>
#include <Q3PopupMenu>
#include <QAction>
#include <QComboBox>
#include <Q3Process>
#include <QTextBrowser>
#include <QUrl>
#else
#include "QSofaLibrary.h"
#include <qheader.h>
#include <qlabel.h>
#include <qlistview.h>
#include <qdragobject.h>
#include <qpushbutton.h>
#include <qtabwidget.h>
#include <qtabbar.h>
#include <qpopupmenu.h>
#include <qaction.h>
#include <qcombobox.h>
#include <qprocess.h>
#include <qtextbrowser.h>
#include <qurl.h>
typedef QProcess Q3Process;
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

typedef sofa::core::ObjectFactory::ClassEntry ClassEntry;
typedef sofa::core::ObjectFactory::Creator    Creator;

using sofa::simulation::tree::GNode;



class SofaModeler : public ::Modeler
{

    Q_OBJECT
public :

    SofaModeler();
    ~SofaModeler() {};

    /// Create a new empty Tab
    void createTab();
    bool closeTab(QWidget *tab, bool forceClose=false);
    /// Change the content of the description box. Happens when the user has clicked on a component
    void changeComponent(const std::string &description);
    void fileOpen(std::string filename);

    /// Change the name of the main window
    void changeNameWindow(std::string filename);

    void changeTabName(GraphModeler *graph, const QString &name, const QString &suffix=QString());
    void setDebugBinary(bool b) {debug=b;};
signals:
    void loadPresetGraph(std::string);


public slots:
    /// Change the state of the Undo button
    void setUndoEnabled(bool v) {this->editUndoAction->setEnabled(v);}
    /// Change the state of the Redo button
    void setRedoEnabled(bool v) {this->editRedoAction->setEnabled(v);}
    /// Each time a graph component is modified, or is cleaned
    void graphModifiedNotification(bool);
    ///Each time a message must be displayed in the status bar (undo/redo, ...)
    void displayMessage(const std::string &m);



    /// Change the content of the description box. Happens when the user has clicked on a component
#ifdef SOFA_QT4
    void changeInformation(Q3ListViewItem *);
#else
    void changeInformation(QListViewItem *);
#endif
    /// Dropping a Node in the Graph
    void newGNode();

    /// Reception of a click on the Sofa library
    void componentDraggedReception( std::string description, std::string categoryName, std::string templateName, ClassEntry* componentEntry);
    /// Build from scratch the Sofa Library
    void rebuildLibrary();
    /// when the GNodeButton is pressed
    void pressedGNodeButton();


    //File Menu
    /// Creation of a new scene (new tab will be created)
    void fileNew() {fileNew(NULL);};
    void fileNew(GNode* root);

    /// Open an existing simulation (new tab will be created)
    void fileOpen();
    void fileOpen(const QString &filename) {fileOpen(std::string(filename.ascii()));}
#ifdef SOFA_QT4
    void fileOpen(const QUrl &filename);
#endif

    /// Save the current simulation
    void fileSave();
    void fileSaveAs();
    void fileReload();

    /// Remove all components of the current simulation
    void clearTab();
    /// Close the current simulation
    void closeTab();
    bool closeTab(int);
    /// Create a new tab containing an empty simulation (by default the collision pipeline is added)
    void newTab();

    /// Quit the Modeler
    void fileExit() {close();};

    void openTutorial();
    /// Launch the current simulation into Sofa
    void runInSofa();
    void runInSofa(const std::string &sceneFilename, GNode *groot);
    void sofaExited();
    void removeTemporaryFiles(const std::string &filename);

    /// Change of simulation by changing the current opened tabulation
    void changeCurrentScene( QWidget*);
    /// Change of simulation by changing the current opened tabulation
    void changeCurrentScene( int n);

    /// Propagate the action Undo to the graph
    void editUndo() {graph->editUndo();}
    /// Propagate the action Redo to the graph
    void editRedo() {graph->editRedo();}


    void editCut();
    void editCopy();
    void editPaste();

    /// Load a preset stored in the menu preset: add a node to the current simulation
    void loadPreset(int);

    /// When the user enter the Modeler, grabbing something: determine the acceptance or not
    void dragEnterEvent( QDragEnterEvent* event);
    /// When the user move the mouse around, with something grabbed
    void dragMoveEvent( QDragMoveEvent* event);

    /// Action to perform when the user drop something in the Modeler
    void dropEvent(QDropEvent* event);

    /// Open a recently Opened files from the menu Recently Opened Files...
    void fileRecentlyOpened(int id);

    /// Filter in the library all the components containing the text written
    void searchText(const FilterQuery&);

    void changeSofaBinary();
    void GUIChanged();
    //When the window is closed: we close all the Sofa launched, and remove temporary files
    void closeEvent ( QCloseEvent * e );

    ///display the plugin manager window, to add/remove some external dynamic libraries
    void showPluginManager();


    void updateViewerList();
protected slots:
    void redirectStderr();
    void redirectStdout();
protected:

    void displayHelpModeler();

    QMenuFilesRecentlyOpened recentlyOpenedFilesManager;

    //********************************************
    //Left Part
    /*           QToolBox     *containerLibrary; */
    SofaLibrary  *library;
    FilterLibrary *filterLibrary;

    //********************************************
    //Right Part
    /// Widget containing all the graphs
    QTabWidget *sceneTab;
    /// Current in-use graph
    GraphModeler *graph; //currentGraph in Use
    /// Current opened Tab
    QWidget *tabGraph;

    SofaTutorialManager *tuto;



    //********************************************
    /// Menu runSofa for the GUI
    Q3PopupMenu *runSofaGUI;
    std::vector< QAction* > listActionGUI;
    /// Menu preset
    Q3PopupMenu *preset;
    /// Menu containing the opened simulations in the Modeler
    Q3PopupMenu *windowMenu;

    QTextBrowser *infoItem;
    /// Correspondance between a name clicked in the menu and a path to the preset
    std::map< Q3PopupMenu*, std::map< std::string, std::string > > mapPreset;


    /// Map between a tabulation from the modeler to an object of type GraphModeler
    std::map<  QWidget*, GraphModeler*> mapGraph;
    /// Map between a tabulation from the modeler to a Sofa Application
    std::multimap<  const QWidget*, Q3Process*> mapSofa;
    /// Map between an index of tabulation to the tabulation itself
    std::map< int, QWidget*> mapWindow;



    /// Is ready to do a paste operation?
    bool isPasteReady;

    /// Number of components currently displayed in the library
    unsigned int displayComponents;

protected slots:
    void editTutorial(const std::string& );


private:
    std::string sofaBinary;
    std::string presetPath;
    std::string examplePath;
    std::string openPath;
    std::string binPath;
    char count;
    std::vector< std::string > exampleFiles;
    bool debug;
};
}
}
}
#endif
