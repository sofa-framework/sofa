/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_MODELER_H
#define SOFA_MODELER_H



#include <QMainWindow>
#include <QToolBar>
#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QHeaderView>
#include <QMenu>
#include <QMenuBar>
#include <QWidget>


#include "GraphModeler.h"
#include "FilterLibrary.h"
#include "SofaTutorialManager.h"
#include <sofa/gui/qt/QDisplayPropertyWidget.h>
#include <sofa/gui/qt/QMenuFilesRecentlyOpened.h>
#include <sofa/gui/qt/SofaPluginManager.h>
#include <sofa/helper/Factory.h>

#include "QSofaTreeLibrary.h"
#include <QTreeWidget>
#include <QDrag>
#include <QPushButton>
#include <QTabWidget>
#include <QTabBar>
#include <QAction>
#include <QComboBox>
#include <QProcess>
#include <QTextBrowser>
#include <QUrl>


namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::simulation::Node;



class SofaModeler: public QMainWindow
{
    Q_OBJECT;
public:
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

	void resizeEvent(QResizeEvent * event);

signals:
    void loadPresetGraph(std::string);


public slots:
    /// Change the state of the Undo button
    void setUndoEnabled(bool v) {this->undoAction->setEnabled(v);}
    /// Change the state of the Redo button
    void setRedoEnabled(bool v) {this->redoAction->setEnabled(v);}
    /// Each time a graph component is modified, or is cleaned
    void graphModifiedNotification(bool);
    ///Each time a message must be displayed in the status bar (undo/redo, ...)
    void displayMessage(const std::string &m);



    /// Change the content of the description box. Happens when the user has clicked on a component
    void changeInformation(QTreeWidgetItem* ,QTreeWidgetItem*);
    /// Dropping a Node in the Graph
    void newGNode();

    /// Reception of a click on the Sofa library
    void componentDraggedReception( std::string description,
                                    std::string categoryName,
                                    std::string templateName,
                                    ClassEntry::SPtr componentEntry);
    /// Build from scratch the Sofa Library
    void rebuildLibrary();
    /// when the GNodeButton is pressed
    void pressedGNodeButton();


    //File Menu
    /// Creation of a new scene (new tab will be created)
    void fileNew() {fileNew(NULL);};
    void fileNew(Node* root);

    /// Open an existing simulation (new tab will be created)
    void fileOpen();
    void fileOpen(const QString &filename) {fileOpen(std::string(filename.toStdString()));}
    void fileOpen(const QUrl &filename);

    /// Save the current simulation
    void fileSave();
    void fileSaveAs();
    void fileReload();

	void exportSofaClasses();

    /// Remove all components of the current simulation
    void clearTab();
    /// Close the current simulation
    void closeTab();
    bool closeTab(int);
    /// Create a new tab containing an empty simulation (by default the collision pipeline is added)
    void newTab();

    /// Quit the Modeler
    void exit() {close();}

    void openTutorial();
    /// Launch the current simulation into Sofa
    void runInSofa();
    void runInSofa(const std::string &sceneFilename, Node *groot);
    void sofaExited(int exitCode, QProcess::ExitStatus status);
    void removeTemporaryFiles(const std::string &filename);

    /// Change of simulation by changing the current opened tabulation
    void changeCurrentScene( int id);
    /// Change of simulation by changing the current opened tabulation
    void changeCurrentScene(QAction *act);

    /// Propagate the action Undo to the graph
    void undo() {graph->undo();}
    /// Propagate the action Redo to the graph
    void redo() {graph->redo();}


    void cut();
    void copy();
    void paste();

    /// Load a preset stored in the menu preset: add a node to the current simulation
    void loadPreset(QAction *act);

    /// When the user enter the Modeler, grabbing something: determine the acceptance or not
    void dragEnterEvent( QDragEnterEvent* event);
    /// When the user move the mouse around, with something grabbed
    void dragMoveEvent( QDragMoveEvent* event);

    /// Action to perform when the user drop something in the Modeler
    void dropEvent(QDropEvent* event);

    /// Open a recently Opened files from the menu Recently Opened Files...
    void fileRecentlyOpened(QAction *act);

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

    // Component Properties
    QDisplayPropertyWidget* propertyWidget;

    // Miscellaneous
    SofaTutorialManager *tuto;



    //********************************************
    /// Menu runSofa for the GUI
    QMenu *runSofaGUI;
    std::vector< QAction* > listActionGUI;
    /// Menu preset
    QMenu *preset;
    /// Menu containing the opened simulations in the Modeler
    QMenu *windowMenu;

    QTextBrowser *infoItem;
    /// Correspondance between a name clicked in the menu and a path to the preset
    std::map< QMenu*, std::map< std::string, std::string > > mapPreset;


    /// Map between a tabulation from the modeler to an object of type GraphModeler
    std::map<  QWidget*, GraphModeler*> mapGraph;
    /// Map between a tabulation from the modeler to a Sofa Application
    std::multimap<  const QWidget*, QProcess*> mapSofa;
    /// Map between an index of tabulation to the tabulation itself
    std::map< QAction*, QWidget*> mapWindow;



    /// Is ready to do a paste operation?
    bool isPasteReady;

    /// Number of components currently displayed in the library
    unsigned int displayComponents;

protected slots:
    void editTutorial(const std::string& );

	// void propertyDockMoved(Q3DockWindow::Place p);


private:
    QWidget *widget;
    QToolBar *toolBar;
    QMenu *fileMenu;
    QMenu *editMenu;

    QAction *newTabAction;
    QAction *closeTabAction;
    QAction *clearTabAction;
    QAction *openAction;
    QAction *saveAction;
    QAction *saveAsAction;
    QAction *reloadAction;
    QAction *exitAction;
    QAction *undoAction;
    QAction *redoAction;
    QAction *cutAction;
    QAction *copyAction;
    QAction *pasteAction;
    QAction *openTutorialsAction;
    QAction *runInSofaAction;
    QAction *openPluginManagerAction;
    QAction *fooAction;
    QAction *exportSofaClassesAction;

    void createActions();
    void createMenu();
    void createToolbar();


private:
    std::string sofaBinary;
    std::string presetPath;
    std::string examplePath;
    std::string openPath;
    std::string binPath;
    char count;
    std::vector< std::string > exampleFiles;
    bool debug;
    SofaPluginManager* plugin_dialog;
};
}
}
}
#endif
