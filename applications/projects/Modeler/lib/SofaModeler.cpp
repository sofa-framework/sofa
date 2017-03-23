/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "SofaModeler.h"

#include <sofa/helper/system/FileRepository.h>

#include <sofa/helper/system/SetDirectory.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <sofa/simulation/XMLPrintVisitor.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/qt/FileManagement.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/cast.h>


#define MAX_RECENTLY_OPENED 10

#include <QToolBox>
#include <QApplication>
#include <QMenuBar>
#include <QMessageBox>
#include <QDir>
#include <QStatusBar>
#include <QDesktopWidget>
#include <QDockWidget>
#include <QVBoxLayout>
#include <QDesktopServices>
#include <QSettings>
#include <QMimeData>

using namespace sofa::core;

namespace sofa
{

namespace gui
{

namespace qt
{


using sofa::helper::system::PluginManager;
using sofa::helper::Utils;


void SofaModeler::createActions()
{

    newTabAction = new QAction(QIcon(":/image0.png"), "New &Tab", this);
    newTabAction->setShortcut(QString("Ctrl+T"));
    connect(newTabAction, SIGNAL(triggered()), this, SLOT(newTab()));

    closeTabAction = new QAction(QIcon(":/imageClose.png"), "&Close Tab", this);
    closeTabAction->setShortcut(QString("Ctrl+W"));
    connect(closeTabAction, SIGNAL(triggered()), this, SLOT(closeTab()));

    clearTabAction = new QAction(QIcon(":/image0.png"), "Clear", this);
    clearTabAction->setShortcut(QString("Ctrl+N"));
    connect(clearTabAction, SIGNAL(triggered()), this, SLOT(clearTab()));

    openAction = new QAction(QIcon(":/image1.png"), "&Open...", this);
    openAction->setShortcut(QString("Ctrl+O"));
    connect(openAction, SIGNAL(triggered()), this, SLOT(fileOpen()));

    saveAction = new QAction(QIcon(":/image3.png"), "&Save", this);
    saveAction->setShortcut(QString("Ctrl+S"));
    connect(saveAction, SIGNAL(triggered()), this, SLOT(fileSave()));

    saveAsAction = new QAction("Save &As", this);
    connect(saveAsAction, SIGNAL(triggered()), this, SLOT(fileSaveAs()));

    reloadAction = new QAction("Reload", this);
    connect(reloadAction, SIGNAL(triggered()), this, SLOT(fileReload()));

    exitAction = new QAction("E&xit", this);
    connect(exitAction, SIGNAL(triggered()), this, SLOT(exit()));

    undoAction = new QAction(QIcon(":/image5.png"), "&Undo", this);
    undoAction->setEnabled(false);
    undoAction->setShortcut(QString("Ctrl+Z"));
    connect(undoAction, SIGNAL(triggered()), this, SLOT(undo()));

    redoAction = new QAction(QIcon(":/image6.png"), "&Redo", this);
    redoAction->setEnabled(false);
    redoAction->setShortcut(QString("Ctrl+Y"));
    connect(redoAction, SIGNAL(triggered()), this, SLOT(redo()));

    cutAction = new QAction(QIcon(":/image7.png"), "&Cut", this);
    cutAction->setShortcut(QString("Ctrl+X"));
    connect(cutAction, SIGNAL(triggered()), this, SLOT(cut()));

    copyAction = new QAction(QIcon(":/image8.png"), "C&opy", this);
    copyAction->setShortcut(QString("Ctrl+C"));
    connect(copyAction, SIGNAL(triggered()), this, SLOT(copy()));

    pasteAction = new QAction(QIcon(":/image9.png"), "&Paste", this);
    pasteAction->setEnabled(false);
    pasteAction->setShortcut(QString("Ctrl+V"));
    connect(pasteAction, SIGNAL(triggered()), this, SLOT(paste()));

    openPluginManagerAction = new QAction("Plugin Manager", this);
    connect(openPluginManagerAction, SIGNAL(triggered()), this, SLOT(showPluginManager()));

    runInSofaAction = new QAction(QIcon(":/image2.png"), "&Run in SOFA", this);
    runInSofaAction->setShortcut(QString("Ctrl+R"));
    connect(runInSofaAction, SIGNAL(triggered()), this, SLOT(runInSofa()));

    openTutorialsAction = new QAction(QIcon(":/image11.png"), "Launch the &Tutorials" ,this);
    connect(openTutorialsAction, SIGNAL(triggered()), this, SLOT(openTutorial()));

    exportSofaClassesAction = new QAction("Export Sofa Classes", this);
    connect(exportSofaClassesAction, SIGNAL(triggered()), this, SLOT(exportSofaClasses()));
}

void SofaModeler::createMenu()
{
    fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction(newTabAction);
    fileMenu->addAction(closeTabAction);
    fileMenu->addSeparator();
    fileMenu->addAction(clearTabAction);
    fileMenu->addAction(openAction);
    fileMenu->addAction(saveAction);
    fileMenu->addAction(saveAsAction);
    fileMenu->addAction(reloadAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exportSofaClassesAction);
    fileMenu->addSeparator();
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    editMenu = menuBar()->addMenu("&Edit");
    editMenu->addAction(undoAction);
    editMenu->addAction(redoAction);
    editMenu->addSeparator();
    editMenu->addAction(cutAction);
    editMenu->addAction(copyAction);
    editMenu->addAction(pasteAction);
    editMenu->addAction(openPluginManagerAction);
}

void SofaModeler::createToolbar()
{
    toolBar = new QToolBar("Toolbar", this);
    toolBar->addAction(newTabAction);
    toolBar->addAction(closeTabAction);
    toolBar->addSeparator();
    toolBar->addAction(clearTabAction);
    toolBar->addAction(openAction);
    toolBar->addAction(saveAction);
    toolBar->addSeparator();
    toolBar->addAction(undoAction);
    toolBar->addAction(redoAction);
    toolBar->addAction(cutAction);
    toolBar->addAction(copyAction);
    toolBar->addAction(pasteAction);
    toolBar->addSeparator();
    addToolBar(Qt::TopToolBarArea, toolBar);
}

SofaModeler::SofaModeler():recentlyOpenedFilesManager(Utils::getSofaPathPrefix() + "/config/Modeler.ini")
    ,runSofaGUI(NULL)
{
    setWindowTitle(QString("Sofa Modeler"));
    setAcceptDrops(true);
    resize(1000, 600);

    createActions();

    fooAction = new QAction("Recently Opened Files...", this);

    createMenu();
    createToolbar();

    widget = new QWidget(this);
    widget->setGeometry(QRect(0, 57, 1000, 543));
    setCentralWidget(widget);


    //index to add in temporary scenes created by the Modeler
    count='0';
//    int menuIndex=4;
    isPasteReady=false;
    pasteAction->setEnabled(false);
    fileMenu->removeAction(fooAction);
    setDebugBinary(false);
    //----------------------------------------------------------------------
    //Get the different path needed
    examplePath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/examples/" );
    openPath = examplePath;
    binPath = Utils::getExecutableDirectory() + "/";
    presetPath = examplePath + std::string("Objects/");
    std::string presetFile = std::string("config/preset.ini" );
    presetFile = sofa::helper::system::DataRepository.getFile ( presetFile );


    QMenu *openTutorial = new QMenu(this);
    openTutorial->setTitle(QString(tr("&Tutorials")));
    this->menuBar()->addMenu(/* QString(tr("&Tutorials")), */ openTutorial /*, menuIndex++ */);
    openTutorial->addAction(openTutorialsAction);
    toolBar->addAction(openTutorialsAction);


    //Find all the scene files in examples directory
    std::vector< QString > exampleQString;
    std::vector< QString > filter;
    const QString path(examplePath.c_str());
    filter.push_back("*.scn"); filter.push_back("*.xml");
    sofa::gui::qt::getFilesInDirectory(path, exampleQString, true, filter);


    //----------------------------------------------------------------------
    // Create the Left part of the GUI
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    //Create a Dock Window to receive the Sofa Library
    QDockWidget *dockLibrary=new QDockWidget("Library", this);
    dockLibrary->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    addDockWidget(Qt::LeftDockWidgetArea, dockLibrary);


    QWidget *leftPartWidget = new QWidget( dockLibrary);
    leftPartWidget->setObjectName("LibraryLayout");
    QVBoxLayout *leftPartLayout = new QVBoxLayout(leftPartWidget);

    //----------------------------------------------------------------------
    //Add a Filter to the Library
    QWidget *filterContainer = new QWidget( leftPartWidget );
    filterContainer->setObjectName("filterContainer");
    QHBoxLayout *filterLayout = new QHBoxLayout(filterContainer);
    leftPartLayout->addWidget(filterContainer);
    //--> Label
    QLabel *filterLabel= new QLabel(QString("Filter: "),filterContainer);
    filterLayout->addWidget(filterLabel);
    //--> Filter
    filterLibrary = new FilterLibrary(filterContainer);
    filterLayout->addWidget(filterLibrary);
    //--> Clear Button
    QPushButton *clearButton= new QPushButton(QString("X"),filterContainer);
    filterLayout->addWidget(clearButton);
    clearButton->setMaximumHeight(16);
    clearButton->setMaximumWidth(16);

    connect(filterLibrary, SIGNAL( filterList( const FilterQuery &) ), this, SLOT(searchText( const FilterQuery &)) );
    connect(clearButton, SIGNAL( clicked()), filterLibrary, SLOT(clearText()));

    //----------------------------------------------------------------------
    //Add the Sofa Library
    QSofaTreeLibrary *l = new QSofaTreeLibrary(leftPartWidget); library = l;
    leftPartLayout->addWidget(l);


    //----------------------------------------------------------------------
    //Add the button to create Node
    QPushButton *GNodeButton = new QPushButton( leftPartWidget);
    GNodeButton->setObjectName("GNodeButton");
    GNodeButton->setText("Node");
    leftPartLayout->addWidget(GNodeButton);
    connect( GNodeButton, SIGNAL( pressed() ),  this, SLOT( pressedGNodeButton() ) );

    dockLibrary->setWidget(leftPartWidget);



    connect(l, SIGNAL( componentDragged( std::string, std::string, std::string, ClassEntry::SPtr) ),
            this, SLOT( componentDraggedReception( std::string, std::string, std::string, ClassEntry::SPtr) ));

    for (unsigned int i=0; i<exampleQString.size(); ++i) exampleFiles.push_back(exampleQString[i].toStdString());

    //----------------------------------------------------------------------
    //Create the information widget
    infoItem = new QTextBrowser(this->centralWidget());
    infoItem->setMaximumHeight(195);
    connect( infoItem, SIGNAL(anchorClicked(const QUrl&)), this, SLOT(fileOpen(const QUrl&)));
#ifndef WIN32
    infoItem->setOpenExternalLinks(true);
#endif
    leftPartLayout->addWidget(infoItem);

    //----------------------------------------------------------------------
    // Create the Right part of the GUI
    //----------------------------------------------------------------------

	// setRightJustification(true);

    QHBoxLayout *mainLayout = new QHBoxLayout(this->centralWidget());

    //----------------------------------------------------------------------
    //Create the scene graph visualization
    sceneTab = new QTabWidget(this->centralWidget());
    mainLayout->addWidget(sceneTab);

    //option available only since Qt 4.5
#if QT_VERSION >= 0x040500
    sceneTab->setTabsClosable(true);
    connect( sceneTab, SIGNAL(tabCloseRequested(int)), this, SLOT(closeTab(int)));
#endif

    connect( sceneTab, SIGNAL(currentChanged( int)), this, SLOT( changeCurrentScene( int)));

    //----------------------------------------------------------------------
    //Create the properties visualization
	ModifyObjectFlags modifyObjectFlags = ModifyObjectFlags();
    modifyObjectFlags.setFlagsForModeler();

	QDockWidget* dockProperty = new QDockWidget("Properties", this);
    dockProperty->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);

    addDockWidget(Qt::RightDockWidgetArea, dockProperty);

	// connect(dockProperty, SIGNAL(placeChanged(Q3DockWindow::Place)), this, SLOT(propertyDockMoved(Q3DockWindow::Place)));
	
    propertyWidget = new QDisplayPropertyWidget(modifyObjectFlags, dockProperty);
	dockProperty->setWidget(propertyWidget);

    //----------------------------------------------------------------------
    //Add plugin manager window. ->load external libs
    plugin_dialog = new SofaPluginManager();
    plugin_dialog->hide();


    this->connect( plugin_dialog->buttonClose, SIGNAL(clicked() ),  this, SLOT( rebuildLibrary() ));
    this->connect( plugin_dialog->buttonClose, SIGNAL(clicked() ),  this, SLOT( updateViewerList() ));

    rebuildLibrary();

    //----------------------------------------------------------------------
    // Create the Menus
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    //Add menu runSofa
    QMenu *runSofaMenu = new QMenu(this);
    runSofaMenu->setTitle(QString(tr("&RunSofa")));
    this->menuBar()->addMenu(runSofaMenu /*, menuIndex++ */);

    runSofaMenu->addAction(runInSofaAction);

    toolBar->addAction(runInSofaAction);

    runSofaMenu->addAction("Change Sofa Binary...", this, SLOT( changeSofaBinary()));
    sofaBinary=std::string();


    runSofaGUI = new QMenu(this);
    runSofaGUI->setTitle(tr("Viewer"));
    runSofaMenu->addMenu(runSofaGUI);

    updateViewerList();



    //----------------------------------------------------------------------
    //Add menu Preset
    preset = new QMenu(this);
    // this->menuBar()->insertItem(tr(QString("&Presets")), preset, menuIndex++);

    //----------------------------------------------------------------------
    //Add menu Window: to quickly find an opened simulation
    windowMenu = new QMenu(this);
    windowMenu->setTitle(QString(tr("&Scenes")));
    //windowMenu->addAction(QIcon(), QString(tr("&Scenes")),this, ));
    this->menuBar()->addMenu(windowMenu);
    windowMenu->addSeparator();
    connect(windowMenu, SIGNAL(triggered(QAction*)), this, SLOT( changeCurrentScene(QAction*)));



    //----------------------------------------------------------------------
    //Create the Preset Menu
    std::multimap< std::string, std::pair< std::string,std::string> > presetArchitecture;

    std::ifstream end(presetFile.c_str());
    std::string s;

    std::string directory, namePreset, nameFile;
    while( std::getline(end,s) )
    {
        //s contains the current line
        std::string::size_type slash = s.find('/');
        if (slash != std::string::npos)
        {
            directory = s; directory.resize(slash);
            s=s.substr(slash+1);

            slash = s.find('/');
            if (slash != std::string::npos)
            {
                namePreset=s; namePreset.resize(slash);
                nameFile = s.substr(slash+1);
                nameFile.resize(nameFile.size()-1);
                presetArchitecture.insert(std::make_pair( directory, std::make_pair( namePreset, nameFile) ) );
            }

        }
    }
    end.close();

    std::multimap< std::string, std::pair< std::string,std::string> >::iterator it_preset = presetArchitecture.begin();
    while(it_preset != presetArchitecture.end())
    {
        std::string directoryName = it_preset->first;
        QMenu* directory = new QMenu(this);
        connect( directory, SIGNAL(triggered(QAction*)), this, SLOT(loadPreset(QAction*)));
        directory->setTitle( tr( it_preset->first.c_str()));
        preset->addMenu(directory);

        std::map< std::string, std::string > &mPreset=mapPreset[directory];

        for (unsigned int i=0; i<presetArchitecture.count(directoryName); i++,it_preset++)
        {
            directory->addAction(it_preset->second.first.c_str());//, this, SLOT(loadPreset()) );
            mPreset.insert(it_preset->second);
        }
    }
    //----------------------------------------------------------------------
    //Configure Recently Opened Menu
//    const int indexRecentlyOpened=fileMenu->actions().count()-2;
    QMenu *recentMenu = recentlyOpenedFilesManager.createWidget(this);
    //TODOQT5
    fileMenu->addMenu(recentMenu /*,indexRecentlyOpened,indexRecentlyOpened */);
    connect(recentMenu, SIGNAL(triggered(QAction*)), this, SLOT(fileRecentlyOpened(QAction*)));


    //----------------------------------------------------------------------
    //Center the application in the screen
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2,  ( screen.height() - this->height()) / 2  );

    //----------------------------------------------------------------------
    //Configure the Tutorials
    tuto=0;
    displayHelpModeler();
}

void SofaModeler::updateViewerList()
{
    //Clear the menu
    std::vector<QAction*>::iterator it;
    for( it = listActionGUI.begin(); it != listActionGUI.end(); ++it)
    {
        runSofaGUI->removeAction(*it);
    }
    listActionGUI.clear();

    //Register all GUIs
    sofa::gui::initMain();

    //Set the different available GUI
    std::vector<std::string> listGUI = sofa::gui::GUIManager::ListSupportedGUI();

    //Insert default GUI
    {
        QAction *act= new QAction(QIcon(), QString("default")+QString("Action"), this);
        act->setText( "default");
        //act->setToggleAction( true );
        act->setCheckable(true);
        act->setChecked(true);
        runSofaGUI->addAction(act);
        listActionGUI.push_back(act);
        connect(act, SIGNAL( triggered()), this, SLOT( GUIChanged() ));
    }
    //Add content of GUI Factory
    for (unsigned int i=0; i<listGUI.size(); ++i)
    {
        QAction *act= new QAction(QIcon(), QString(listGUI[i].c_str())+QString("Action"), this);
        act->setText( QString(listGUI[i].c_str()));
        act->setCheckable(true);
        runSofaGUI->addAction(act);
        listActionGUI.push_back(act);
        connect(act, SIGNAL( triggered()), this, SLOT( GUIChanged() ));
    }

}



void SofaModeler::rebuildLibrary()
{
    library->clear();
    library->build(exampleFiles);

}

void SofaModeler::closeEvent( QCloseEvent *e)
{
    const int numTab=sceneTab->count();
    bool closeProgram=true;
    for (int i=numTab-1; i>=0; --i) closeProgram &=closeTab(i);
    if  (closeProgram) e->accept();
    else e->ignore();
}

void SofaModeler::fileNew( Node* root)
{
    if (!root) graph->setFilename("");
    changeNameWindow("");
    //no parent, adding root: if root is NULL, then an empty Node will be created
    graph->setRoot( root, false);
    sceneTab->setCurrentIndex(sceneTab->count()-1);
}

void SofaModeler::fileOpen()
{
    QString s = getOpenFileName ( this, QString(openPath.c_str()),"Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;All (*)", "open file dialog",  "Choose a file to open" );
    if (s.length() >0)
    {
        fileOpen(s);
    }
}

void SofaModeler::clearTab()
{
    if (graph)	fileNew();
}


void SofaModeler::newTab()
{
    std::string newScene="config/newScene.scn";
    if (sofa::helper::system::DataRepository.findFile(newScene))
    {
        std::string openPathPrevious = openPath;
        newScene = sofa::helper::system::DataRepository.getFile ( newScene);
        fileOpen(newScene);
        graph->setFilename("");
        openPath = openPathPrevious;
    }
    else
    {
        createTab();
    }
    displayHelpModeler();
}

void SofaModeler::createTab()
{
    QWidget *newtab = new QWidget();
    tabGraph = newtab;
    QVBoxLayout *currentTabLayout = new QVBoxLayout(newtab);
    currentTabLayout->setObjectName(QString("ModelerScene"));
    sceneTab->addTab(newtab, QString("New Scene"));
    GraphModeler* modelerGraph = new GraphModeler(newtab,"Modeler");
    mapGraph.insert(std::make_pair(newtab, modelerGraph));
    mapGraph[newtab] = modelerGraph;
    graph = modelerGraph;

    graph->setAcceptDrops(true);
    currentTabLayout->addWidget(graph,0,0);

    graph->setSofaLibrary(library);
    graph->setPropertyWidget(propertyWidget);
    graph->setPreset(preset);
    fileNew();

    connect(graph, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(changeInformation(QTreeWidgetItem *,QTreeWidgetItem*)));
    connect(graph, SIGNAL( fileOpen(const QString&)), this, SLOT(fileOpen(const QString&)));
    connect(graph, SIGNAL( undoEnabled(bool)), this, SLOT(setUndoEnabled(bool)));
    connect(graph, SIGNAL( redoEnabled(bool)), this, SLOT(setRedoEnabled(bool)));
    connect(graph, SIGNAL( graphModified(bool)), this, SLOT(graphModifiedNotification(bool)));
    connect(graph, SIGNAL( displayMessage(const std::string&)), this, SLOT(displayMessage(const std::string &)));
}

void SofaModeler::closeTab()
{
    if (sceneTab->count() == 0) return;
    closeTab(tabGraph);
}

bool SofaModeler::closeTab(int i)
{
    if (i<0) return true;
    return closeTab(sceneTab->widget(i));
}

bool SofaModeler::closeTab(QWidget *curTab, bool forceClose)
{
    GraphModeler *mod = mapGraph[curTab];
    if (!forceClose && mod->isUndoEnabled()) //means modifications have been performed
    {

        const QString caption("Unsaved Modifications Notification");
        QString warning=QString("The current scene ")+ QString(sofa::helper::system::SetDirectory::GetFileName(mod->getFilename().c_str()).c_str()) + QString(" has been modified, do you want to save it?");
        int response=QMessageBox::warning ( this, caption,warning,QMessageBox::No, QMessageBox::Ok,  QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape);
        if ( response == QMessageBox::Cancel )
            return false;
        else if (response == QMessageBox::Ok)
        {
            if (mod->getFilename().empty()) fileSaveAs();
            else simulation::tree::getSimulation()->exportXML(mod->getRoot(), mod->getFilename().c_str());
        }
    }
    //If the scene has been launch in Sofa
    if (mapSofa.size() &&
        mapSofa.find(curTab) != mapSofa.end())
    {
        typedef std::multimap< const QWidget*, QProcess* >::iterator multimapIterator;
        std::pair< multimapIterator,multimapIterator > range;
        range=mapSofa.equal_range(curTab);
        for (multimapIterator it=range.first; it!=range.second; ++it)
        {
            removeTemporaryFiles(it->second->objectName().toStdString());
            it->second->kill();
        }
        mapSofa.erase(range.first, range.second);
    }

    //Find the scene in the window menu
    std::map< QAction*, QWidget* >::const_iterator it;
    for (it = mapWindow.begin(); it!=mapWindow.end(); ++it)
    {
        if (it->second == curTab) break;
    }
    windowMenu->removeAction(it->first);
    mapWindow.erase(it->first);

    //Closing the Modify Dialog opened
    if (dynamic_cast< GraphModeler* >(mod))
    {
        mod->closeDialogs();
        mod->close();
    }


    sceneTab->removeTab(sceneTab->indexOf(curTab));
    mapGraph.erase(curTab);
    curTab->close();
    return true;
}

void SofaModeler::fileOpen(const QUrl &u)
{
#ifdef WIN32
    if(u.toString().startsWith("http"))
    {
        QDesktopServices::openUrl(u);
    }
    else
    {
        std::string path=u.toString().toStdString();
        fileOpen(path);
    }
#else
    std::string path=u.path().toStdString();
    fileOpen(path);
#endif
}


void SofaModeler::fileOpen(std::string filename)
{
    if ( sofa::helper::system::DataRepository.findFile ( filename ) )
    {
        filename =  sofa::helper::system::DataRepository.getFile ( filename );
        openPath = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
        Node::SPtr root = NULL;
        root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(filename.c_str()).get() );
        if (root)
        {
            createTab();
            fileNew(root.get());
            sceneTab->setCurrentIndex(sceneTab->count()-1);

            graph->setFilename(filename);
            changeTabName(graph,QString(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).c_str()));

            changeNameWindow(graph->getFilename());
            QAction* act = windowMenu->addAction(graph->getFilename().c_str());
            mapWindow.insert(std::make_pair(act, tabGraph));
            recentlyOpenedFilesManager.openFile(filename);
        }
    }
    displayHelpModeler();
}

void SofaModeler::fileRecentlyOpened(QAction* act)
{
    fileOpen(act->text());
}

void SofaModeler::pressedGNodeButton()
{
    newGNode();
    QPushButton *button = (QPushButton*) sender();
    button->setDown(false);
}

void SofaModeler::fileSave()
{
    if (sceneTab->count() == 0) return;
    if (graph->getFilename().empty()) fileSaveAs();
    else 	                          graph->save(graph->getFilename());
}


void SofaModeler::fileSaveAs()
{
    if (sceneTab->count() == 0) return;
    std::string path;
    if (graph->getFilename().empty()) path=examplePath.c_str();
    else path=sofa::helper::system::SetDirectory::GetParentDir(graph->getFilename().c_str());
    QString s = sofa::gui::qt::getSaveFileName ( this, QString(path.c_str()), "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
    {

        std::string extension=sofa::helper::system::SetDirectory::GetExtension(s.toStdString().c_str());
        if (extension.empty()) s+=QString(".scn");

        graph->save( s.toStdString() );
        //  	    if (graph->getFilename().empty())
        //  	      {
        std::string filename = s.toStdString();
        graph->setFilename(filename);
        changeNameWindow(filename);
        changeTabName(graph, QString(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).c_str()));
        examplePath = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
        //  	      }
        recentlyOpenedFilesManager.openFile(filename);
    }
}

void SofaModeler::fileReload()
{
    if (sceneTab->count() == 0) return;
    if (graph->getFilename().empty()) return;
    else
    {
        const std::string filename=graph->getFilename();
        closeTab(tabGraph,true);
        fileOpen(filename);
    }
}

void SofaModeler::exportSofaClasses()
{
	QString filename = sofa::gui::qt::getSaveFileName(this, QString(binPath.c_str()), "Sofa Classes (*.xml)", "export classes dialog", "Choose where the Sofa classes will be exported");
	if(filename.isEmpty())
		return;

	simulation::Node::SPtr root = getSimulation()->createNewGraph("components");

	std::vector< ClassEntry::SPtr > entries;
	sofa::core::ObjectFactory::getInstance()->getAllEntries(entries);
	//Set of categories found in the Object Factory
	std::set< std::string > mainCategories;
	//Data containing all the entries for a given category
	std::multimap< std::string, ClassEntry::SPtr > inventory;

	for (std::size_t i=0; i<entries.size(); ++i)
	{
		//Insert Template specification
		std::vector< std::string >::iterator it;
		std::vector<std::string> categories;
		sofa::core::CategoryLibrary::getCategories(entries[i]->creatorMap.begin()->second->getClass(), categories);
		for (it = categories.begin(); it != categories.end(); ++it)
		{
			mainCategories.insert((*it));
			inventory.insert(std::make_pair((*it), entries[i]));
		}
	}

	std::set< std::string >::iterator itCategory;
	typedef std::multimap< std::string, ClassEntry::SPtr >::iterator IteratorInventory;

	//We add the components category by category
	for (itCategory = mainCategories.begin(); itCategory != mainCategories.end(); ++itCategory)
	{
		const std::string& categoryName = *itCategory;

		std::pair< IteratorInventory,IteratorInventory > rangeCategory;
		rangeCategory = inventory.equal_range(categoryName);

		simulation::Node::SPtr node = root->createChild(categoryName);

		//Process all the component of the current category, and add them to the group
		for (IteratorInventory itComponent=rangeCategory.first; itComponent != rangeCategory.second; ++itComponent)
		{
			ObjectFactory::ClassEntry::SPtr entry = itComponent->second;
			const std::string &componentName = entry->className;

			if(0 == entry->className.compare("Distances")) // this component lead to a crash
				continue;

			int componentCount = 0;
			for(ObjectFactory::CreatorMap::const_iterator creatorIterator = entry->creatorMap.begin(); creatorIterator != entry->creatorMap.end(); ++creatorIterator)
			{
				const std::string& templateName = creatorIterator->first;

				std::stringstream componentNameStream;
				componentNameStream << componentName << componentCount++;

				BaseObjectDescription desc(componentNameStream.str().c_str(), componentName.c_str());
				desc.setAttribute("template", templateName.c_str());

				// print log
				//std::cout << componentName << " - " << templateName <<  std::endl;

				if(creatorIterator->second->canCreate(node->getContext(), &desc))
					creatorIterator->second->createInstance(node->getContext(), &desc);
				else
					creatorIterator->second->createInstance(node->getContext(), 0);
			}
		}
	}

	getSimulation()->exportXML(root.get(), filename.toStdString().c_str());

	std::cout << "Sofa classes have been XML exported in: " << filename.toStdString() << std::endl << std::endl;
}

void SofaModeler::loadPreset(QAction* act)
{
    QMenu *s = (QMenu*) sender();
    const std::string elementClicked(act->text().toStdString());
    std::string presetFile = presetPath+ mapPreset[s][elementClicked];

    if (sofa::helper::system::DataRepository.findFile ( presetFile ))
    {
        presetFile = sofa::helper::system::DataRepository.getFile ( presetFile );
        graph->loadPreset(presetFile);
    }
    else std::cerr<<"Preset : " << presetFile << " Not found\n";
}

void SofaModeler::changeTabName(GraphModeler *graph, const QString &name, const QString &suffix)
{
    QWidget *tabGraph=0;
    QString fullPath(graph->getFilename().c_str());
    if (fullPath.isEmpty())
    {
        fullPath = QString(sofa::helper::system::DataRepository.getFile("config/newScene.scn").c_str());
    }
    //Update the name of the tab
    {
        std::map<QWidget*, GraphModeler*>::iterator it;


        for (it=mapGraph.begin(); it!=mapGraph.end(); ++it)
        {
            if (it->second == graph)
            {
                sceneTab->setTabText(sceneTab->indexOf(it->first), name+suffix);
                sceneTab->setTabToolTip(sceneTab->indexOf(tabGraph), fullPath+suffix);

                tabGraph = it->first;
                break;
            }
        }
    }

    if (!tabGraph) return;

    //Update the Scene menu
    {
        std::map< QAction*, QWidget *>::iterator it;
        for (it=mapWindow.begin(); it!=mapWindow.end(); ++it)
        {
            if (it->second == tabGraph)
            {
                it->first->setText(fullPath + suffix);
                break;
            }
        }
    }
}

void SofaModeler::resizeEvent(QResizeEvent * /*event*/)
{
	// Q3DockArea* dockArea = rightDock();
	// if(dockArea)
	// {
	// 	QList<Q3DockWindow*> dockWindowList = dockArea->dockWindowList();

	// 	int width = event->size().width();
	// 	if(width < 1200)
	// 		width /= 4;
	// 	else
	// 		width /= 3;

	// 	while(!dockWindowList.isEmpty())
	// 	{
	// 		Q3DockWindow* dockWindow = dockWindowList.takeFirst();
	// 		dockWindow->setFixedExtentWidth(width);
	// 	}

	// 	update();
	// }
}

void SofaModeler::graphModifiedNotification(bool modified)
{
    GraphModeler *graph = (GraphModeler*) sender();
    QString suffix;
    if (modified) suffix = QString("*");

    QString tabName;
    if (graph->getFilename().empty()) tabName=QString("newScene.scn");
    else tabName=QString(sofa::helper::system::SetDirectory::GetFileName(graph->getFilename().c_str()).c_str());

    changeTabName(graph, tabName, suffix);
}


void SofaModeler::changeInformation(QTreeWidgetItem *item, QTreeWidgetItem * /* previous */)
{
    if (!item || item->childCount() != 0)
    {
        displayHelpModeler();
        return;
    }
    std::string nameObject = item->text(0).toStdString();
    std::string::size_type end_name = nameObject.find(" ");
    if (end_name != std::string::npos) nameObject.resize(end_name);
    changeComponent( library->getComponentDescription(nameObject) );
}


void SofaModeler::componentDraggedReception( std::string description, std::string // categoryName
        , std::string templateName, ClassEntry::SPtr componentEntry)
{
    changeComponent(description );
    if (!graph) return;
    graph->setLastSelectedComponent(templateName, componentEntry);
    if (tuto && tuto->isVisible()) tuto->getGraph()->setLastSelectedComponent(templateName, componentEntry);

    QDrag *dragging = new QDrag((QWidget*) this->sender());
    QMimeData* mimedata = new QMimeData();
    mimedata->setText("ComponentCreation");
    dragging->setMimeData(mimedata);
    dragging->exec(Qt::CopyAction | Qt::MoveAction);
    //dragging->dragCopy();
}

void SofaModeler::changeComponent(const std::string &description)
{
    infoItem->setText(description.c_str());
}


void SofaModeler::newGNode()
{
    QDrag *dragging = new QDrag((QPushButton*) this->sender());
    QMimeData* mimedata = new QMimeData();
    mimedata->setText("Node");
    dragging->setMimeData(mimedata);

    //dragging->dragCopy();
    dragging->exec(Qt::CopyAction | Qt::MoveAction);
}


void SofaModeler::changeCurrentScene( int id)
{
    QWidget* currentGraph = sceneTab->widget(id);

    tabGraph=currentGraph;

    graph = mapGraph[currentGraph];
    if (graph)
    {
        changeNameWindow(graph->getFilename());
        undoAction->setEnabled(graph->isUndoEnabled());
        redoAction->setEnabled(graph->isRedoEnabled());
    }
    else
    {
        undoAction->setEnabled(false);
        redoAction->setEnabled(false);
    }
}


void SofaModeler::changeCurrentScene(QAction* act)
{
    sceneTab->setCurrentWidget(mapWindow[act] );
}


void SofaModeler::changeNameWindow(std::string filename)
{

    std::string str = "Sofa Modeler";
    if (!filename.empty()) str+= std::string(" - ") + filename;

    setWindowTitle ( str.c_str() );

}

void SofaModeler::dropEvent(QDropEvent* event)
{
    QPushButton *push = (QPushButton *)event->source();
    if (push) push->setDown(false);

    QString text;

    if (event->mimeData()->hasText())
        text = event->mimeData()->text();

    std::string filename(text.toStdString());
    std::string test = filename; test.resize(4);
    if (test == "file")
    {
#ifdef WIN32
        filename = filename.substr(8); //removing file:///
#else
        filename = filename.substr(7); //removing file://
#endif

        if (filename[filename.size()-1] == '\n')
        {
            filename.resize(filename.size()-1);
            filename[filename.size()-1]='\0';
        }

        fileOpen(filename);
    }
}

void SofaModeler::editTutorial(const std::string& filename)
{
    std::string tutorialFilename(filename);
    fileOpen(tutorialFilename);
    //this->setActiveWindow();
}

// void SofaModeler::propertyDockMoved(Q3DockWindow::Place p)
// {
// 	Q3DockWindow* dockWindow = qobject_cast<Q3DockWindow*>(sender());
// 	if(!dockWindow)
// 		return;

// 	if(Q3DockWindow::OutsideDock == p)
// 		dockWindow->resize(500, 700);
// }

void SofaModeler::openTutorial()
{
    if (tuto)
    {
        if (tuto->isVisible()) return;
        delete tuto;
    }

    tuto=new SofaTutorialManager(this, "tutorial");
    connect(tuto, SIGNAL(runInSofa(const std::string&, Node*)), this, SLOT(runInSofa(const std::string&, Node*)));
    connect(tuto, SIGNAL(editInModeler(const std::string&)), this, SLOT(editTutorial(const std::string& ) ));
    GraphModeler *graphTuto=tuto->getGraph();
    graphTuto->setSofaLibrary(library);
    graphTuto->setPreset(preset);
    connect(graphTuto, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(changeInformation(QTreeWidgetItem *,QTreeWidgetItem*)));

    tuto->show();
}

void SofaModeler::runInSofa()
{
    if (sceneTab->count() == 0) return;
    Node* root=graph->getRoot();
    runInSofa(graph->getFilename(), root);
}
void SofaModeler::runInSofa(	const std::string &sceneFilename, Node* root)
{
    if (!root) return;
    // Init the scene
    sofa::gui::GUIManager::Init("Modeler");

    //Saving the scene in a temporary file ==> doesn't modify the current Node of the simulation
    std::string path;
    if (sceneFilename.empty()) path=presetPath;
    else path = sofa::helper::system::SetDirectory::GetParentDir(sceneFilename.c_str())+std::string("/");


    std::string filename=path + std::string("temp") + (++count) + std::string(".scn");
    simulation::tree::getSimulation()->exportXML(root,filename.c_str());

    //Make a copy of the .view if it exists for the current viewer
    const std::string &extension=sofa::helper::system::SetDirectory::GetExtension(sceneFilename.c_str());

    if (!sceneFilename.empty() && !extension.empty())
    {

        std::string viewFile = sceneFilename;
        //Get the name of the viewer
        std::string viewerName;
        for (unsigned int i=0; i<listActionGUI.size(); ++i)
        {
            if (listActionGUI[i]->isChecked())
            {
                viewerName = listActionGUI[i]->text().toStdString();
                if (viewerName == "default")
                    viewerName = sofa::gui::GUIManager::GetValidGUIName();

                if (viewerName == "qt") //default viewer: no extension
                {
                    viewerName.clear();
                }
                break;
            }
        }

        std::string viewerExtension;
        if (!viewerName.empty())
            viewerExtension += "." + viewerName;
        viewerExtension += ".view";

        viewFile += viewerExtension;

        if ( !sofa::helper::system::DataRepository.findFile ( viewFile ) )
        {
            viewFile = sceneFilename+".view";
            viewerExtension = ".view";
        }

        //        std::cerr << "viewFile = " << viewFile << std::endl;
        if ( sofa::helper::system::DataRepository.findFile ( viewFile ) )
        {
            std::ifstream originalViewFile(viewFile.c_str());
            const std::string nameCopyViewFile(path + std::string("temp") + count + ".scn" + viewerExtension );
            std::ofstream copyViewFile(nameCopyViewFile.c_str());
            std::string line;
            while (std::getline(originalViewFile, line)) copyViewFile << line << "\n";

            copyViewFile.close();
            originalViewFile.close();
        }
    }
    if (count > '9') count = '0';

    QString messageLaunch;
    QStringList argv;
    //=======================================
    // Run Sofa
    if (sofaBinary.empty()) //If no specific binary is specified, we use runSofa
    {
        std::string binaryName="runSofa";
#ifndef NDEBUG
        binaryName+="_d";
#endif

#ifdef WIN32
        sofaBinary = binPath + binaryName + ".exe";
#else
        sofaBinary = binPath + binaryName;
#endif
    }

    argv << QString(filename.c_str());

    messageLaunch = QString("Use command: ")
            + QString(sofaBinary.c_str())
            + QString(" ");

    //Setting the GUI
    for (unsigned int i=0; i<listActionGUI.size(); ++i)
    {
        if (listActionGUI[i]->isChecked())
        {
            if (std::string(listActionGUI[i]->text().toStdString()) != "default")
            {
                argv << "-g" << listActionGUI[i]->text();
                messageLaunch += QString("-g ") + QString(listActionGUI[i]->text());
            }
            break;
        }
    }

    //retrive plugins
    typedef sofa::helper::system::PluginManager::PluginMap PluginMap;
    PluginMap& pluginMap = PluginManager::getInstance().getPluginMap();
    PluginManager::PluginIterator it;

    for( it = pluginMap.begin(); it != pluginMap.end(); ++it )
    {
        argv << "-l" << QString((it->first).c_str()) << " ";
        messageLaunch += QString("-l ") + QString((it->first).c_str());
    }


    argv << "-t";


    QProcess *p = new QProcess(this);


    p->setWorkingDirectory(QString(binPath.c_str()) );
    p->setObjectName(QString(filename.c_str()) );


    connect(p, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(sofaExited(int, QProcess::ExitStatus)));
    QDir dir(QString(sofa::helper::system::SetDirectory::GetParentDir(sceneFilename.c_str()).c_str()));
    connect(p, SIGNAL( readyReadStandardOutput () ), this , SLOT ( redirectStdout() ) );
    connect(p, SIGNAL( readyReadStandardError () ), this , SLOT ( redirectStderr() ) );

    p->start(QString(sofaBinary.c_str()), argv);

    mapSofa.insert(std::make_pair(tabGraph, p));

    statusBar()->showMessage(messageLaunch,5000);
}

void SofaModeler::redirectStdout()
{
    QProcess* p = ((QProcess*) sender());
    if( !p )
    {
        return;
    }

    if (p->waitForStarted(-1))
    {
        std::cout << "FROM SOFA [OUT] >> " << QString(p->readAllStandardOutput()).toStdString() << std::endl;
    }
}

void SofaModeler::redirectStderr()
{
    QProcess* p = ((QProcess*) sender());
    if( !p )
    {
        return;
    }
    if (p->waitForStarted(-1))
    {
        std::cerr << "FROM SOFA [ERR] >> " << QString(p->readAllStandardError()).toStdString() << std::endl;
    }
}

void SofaModeler::sofaExited(int exitCode, QProcess::ExitStatus existStatus)
{
    QProcess *p = ((QProcess*) sender());
    std::string programName;

    programName = p->objectName().toStdString();

    removeTemporaryFiles(programName);
    if (existStatus == QProcess::NormalExit )
    {
        p->closeWriteChannel();
        disconnect(p, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(sofaExited(int, QProcess::ExitStatus)));
        disconnect(p, SIGNAL( readyReadStandardOutput () ), this , SLOT ( redirectStdout() ) );
        disconnect(p, SIGNAL( readyReadStandardError () ), this , SLOT ( redirectStderr() ) );
        if(p->atEnd())
            std::cout << "Sofa exited safely." << std::endl;
        else
            std::cout << "Chelou." << std::endl;
        p->kill();
        return;
    }
    typedef std::multimap< const QWidget*, QProcess* >::iterator multimapIterator;
    for (multimapIterator it=mapSofa.begin(); it!=mapSofa.end(); ++it)
    {
        if (it->second == p)
        {
            const QString caption("Problem");
            const QString warning("Error running Sofa, error code ");
            QMessageBox::critical( this, caption,warning + QString(exitCode), QMessageBox::Ok | QMessageBox::Escape, QMessageBox::NoButton );
            return;
        }
    }

}

void SofaModeler::removeTemporaryFiles(const std::string &f)
{
    std::string filename(f);
    std::string copyBuffer(presetPath+"copyBuffer.scn");
    //Delete Temporary file
    ::remove(filename.c_str());

    //Delete View files
    std::string viewFilename=filename + std::string(".view");
    for (unsigned int i=0; i<listActionGUI.size(); ++i)
    {
        const std::string viewerName=listActionGUI[i]->text().toStdString();
        if (viewerName != "default" && viewerName != "batch")
        {
            std::string viewFilename=filename + std::string(".") + viewerName + std::string(".view");
            //Remove eventual .view file
            ::remove(viewFilename.c_str());
        }
    }

    //Remove eventual copy buffer
    ::remove(copyBuffer.c_str());
}


/// When the user enter the Modeler, grabbing something: determine the acceptance or not
void SofaModeler::dragEnterEvent( QDragEnterEvent* event)
{
    QString text;
    if (event->mimeData()->hasText())
        text = event->mimeData()->text();
    if (text.isEmpty()) event->ignore();
    else
    {
        std::string filename(text.toStdString());
        std::string test = filename; test.resize(4);
        if (test == "file")  event->accept();
        else event->ignore();
    }
}

/// When the user move the mouse around, with something grabbed
void SofaModeler::dragMoveEvent( QDragMoveEvent* event)
{
    QString text;
    if (event->mimeData()->hasText())
        text = event->mimeData()->text();
    if (text.isEmpty()) event->ignore();
    else
    {
        std::string filename(text.toStdString());
        std::string test = filename; test.resize(4);
        if (test == "file")  event->accept();
        else event->ignore();
    }
}


/// Quick Filter of the components
void SofaModeler::searchText(const FilterQuery& query)
{
    QSofaTreeLibrary* l=static_cast<QSofaTreeLibrary*>(library);
    l->filter(query);
}


/*****************************************************************************************************************/
//runSofa Options
void SofaModeler::changeSofaBinary()
{

    QString s = getOpenFileName ( this, QString(binPath.c_str()),
#ifdef WIN32
            "All Files(*.exe)",
#else
            "All Files(*)",
#endif
            "open sofa binary",  "Choose a binary to use" );
    if (s.length() >0)
    {
        sofaBinary=s.toStdString();
        binPath=sofa::helper::system::SetDirectory::GetParentDir(sofaBinary.c_str());
    }
    else
    {
        sofaBinary.clear();
    }
}

void SofaModeler::GUIChanged()
{
    QAction *act=(QAction*) sender();
    for (unsigned int i=0; i<listActionGUI.size(); ++i)
    {
        listActionGUI[i]->setChecked(listActionGUI[i] == act);
    }
}

/*****************************************************************************************************************/
//Cut/Copy Paste management
void SofaModeler::cut()
{
    if (graph)
    {
        isPasteReady=graph->cut(presetPath+"copyBuffer.scn");
        pasteAction->setEnabled(isPasteReady);
    }
}
void SofaModeler::copy()
{
    if (graph)
    {
        isPasteReady=graph->copy(presetPath+"copyBuffer.scn");
        pasteAction->setEnabled(isPasteReady);
    }
}
void SofaModeler::paste()
{
    if (graph)
    {
        graph->paste(presetPath+"copyBuffer.scn");
    }
}

void SofaModeler::showPluginManager()
{
    plugin_dialog->show();
}

void SofaModeler::displayMessage(const std::string &m)
{
    QString messageLaunch(m.c_str());
    statusBar()->showMessage(messageLaunch,5000);
}

void SofaModeler::displayHelpModeler()
{
    static std::string pathModelerHTML=sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/applications/projects/Modeler/Modeler.html" );
#ifdef WIN32
    infoItem->setSource(QUrl(QString("file:///")+QString(pathModelerHTML.c_str())));
#else
    infoItem->setSource(QUrl(QString(pathModelerHTML.c_str())));
#endif
}


}
}
}
