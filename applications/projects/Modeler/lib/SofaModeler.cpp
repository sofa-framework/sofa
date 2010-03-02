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
#include "SofaModeler.h"

#include <sofa/helper/system/FileRepository.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/tree/TreeSimulation.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/qt/FileManagement.h>
#include <sofa/gui/qt/SofaPluginManager.h>

#define MAX_RECENTLY_OPENED 10


#include <map>
#include <set>
#include <cstdio>


//Automatically create and destroy all the components available: easy way to verify the default constructor and destructor
//#define TEST_CREATION_COMPONENT

#ifdef SOFA_QT4
#include <QToolBox>
#include <QSpacerItem>
#include <QGridLayout>
#include <QTextEdit>
#include <Q3TextBrowser>
#include <QLabel>
#include <QApplication>
#include <QMenuBar>
#include <QMessageBox>
#include <QDir>
#include <QStatusBar>
#include <QDesktopWidget>
#include <Q3DockWindow>
#include <Q3DockArea>
#else
#include <qtoolbox.h>
#include <qlayout.h>
#include <qtextedit.h>
#include <qtextbrowser.h>
#include <qapplication.h>
#include <qmenubar.h>
#include <qmessagebox.h>
#include <qdir.h>
#include <qstatusbar.h>
#include <qdockwindow.h>
#include <qdockarea.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QTextDrag Q3TextDrag;
typedef QDockWindow Q3DockWindow;
#endif


SofaModeler::SofaModeler()
{
    //index to add in temporary scenes created by the Modeler
    count='0';
    isPasteReady=false;
    editPasteAction->setEnabled(false);
#ifdef SOFA_QT4
    fileMenu->removeAction(Action);
#endif


    //----------------------------------------------------------------------
    //Get the different path needed
    examplePath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/examples/" );
    openPath = examplePath;
    binPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/bin/" );
    presetPath = examplePath + std::string("Objects/");
    std::string presetFile = std::string("config/preset.ini" );
    presetFile = sofa::helper::system::DataRepository.getFile ( presetFile );



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
    Q3DockWindow *dockRecorder=new Q3DockWindow(this);
    dockRecorder->setResizeEnabled(true);
    this->moveDockWindow( dockRecorder, Qt::DockLeft);
//        this->topDock() ->setAcceptDockWindow(dockRecorder,false);
//        this->bottomDock()->setAcceptDockWindow(dockRecorder,false);
    dockRecorder->setFixedExtentWidth(520);
    QWidget *leftPartWidget = new QWidget( dockRecorder, "LibraryLayout");
    QVBoxLayout *leftPartLayout = new QVBoxLayout(leftPartWidget);

    //----------------------------------------------------------------------
    //Add a Filter to the Library
    QWidget *filterContainer = new QWidget( leftPartWidget, "filterContainer" );
    QHBoxLayout *filterLayout = new QHBoxLayout(filterContainer);
    leftPartLayout->addWidget(filterContainer);
    //--> Label
    QLabel *filterLabel = new QLabel(filterContainer, "filterLabel");
    filterLabel->setText( "Filter: " );
    filterLayout->addWidget(filterLabel);
    //--> Filter
    filterLibrary = new FilterLibrary(filterContainer);
    filterLayout->addWidget(filterLibrary);
    connect(filterLibrary, SIGNAL( filterList( const FilterQuery &) ), this, SLOT(searchText( const FilterQuery &)) );

    //----------------------------------------------------------------------
    //Add the Sofa Library
#ifdef SOFA_QT4
    QSofaTreeLibrary *l = new QSofaTreeLibrary(leftPartWidget); library = l;
#else
    QSofaLibrary *l = new QSofaLibrary(leftPartWidget); library = l;
#endif
    leftPartLayout->addWidget(l);


    //----------------------------------------------------------------------
    //Add the button to create GNode
    QPushButton *GNodeButton = new QPushButton( leftPartWidget, "GNodeButton");
    GNodeButton->setText("GNode");
    leftPartLayout->addWidget(GNodeButton);
    connect( GNodeButton, SIGNAL( pressed() ),  this, SLOT( pressedGNodeButton() ) );

    dockRecorder->setWidget(leftPartWidget);



    connect(l, SIGNAL( componentDragged( std::string, std::string, std::string, ClassEntry *) ),
            this, SLOT( componentDraggedReception( std::string, std::string, std::string, ClassEntry *) ));

    for (unsigned int i=0; i<exampleQString.size(); ++i) exampleFiles.push_back(exampleQString[i].ascii());


    //----------------------------------------------------------------------
    // Create the Right part of the GUI
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    //Create the scene graph visualization
    QWidget *GraphSupport = new QWidget(this->centralWidget());
    QGridLayout* GraphLayout = new QGridLayout(GraphSupport, 1,1,5,2,"GraphLayout");
    sceneTab = new QTabWidget(GraphSupport);
    GraphLayout->addWidget(sceneTab,0,0);

#ifdef SOFA_QT4

    //option available only since Qt 4.5
#if QT_VERSION >= 0x040500
    sceneTab->setTabsClosable(true);
    connect( sceneTab, SIGNAL(tabCloseRequested(int)), this, SLOT(closeTab(int)));
#endif

#endif

    connect( sceneTab, SIGNAL(currentChanged( QWidget*)), this, SLOT( changeCurrentScene( QWidget*)));

    GraphSupport->resize(200,550);

    this->centralWidget()->layout()->add(GraphSupport);

    //----------------------------------------------------------------------
    //Add plugin manager window. ->load external libs
    SofaPluginManager::getInstance()->hide();
    this->connect( SofaPluginManager::getInstance()->buttonClose, SIGNAL(clicked() ),  this, SLOT( rebuildLibrary() ));

    library->build(exampleFiles);
    int menuIndex=4;

    //----------------------------------------------------------------------
    // Create the Menus
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    //Add menu runSofa
    Q3PopupMenu *runSofaMenu = new Q3PopupMenu(this);
    this->menubar->insertItem(tr(QString("&RunSofa")), runSofaMenu, menuIndex++);

    runInSofaAction->addTo(runSofaMenu);

    runInSofaAction->addTo(toolBar);

    runSofaMenu->insertItem("Change Sofa Binary...", this, SLOT( changeSofaBinary()));
    sofaBinary=std::string();

    Q3PopupMenu *runSofaGUI = new Q3PopupMenu(this);
    runSofaMenu->insertItem(QIconSet(), tr("Viewer"), runSofaGUI);

    //Set the different available GUI
    std::vector<std::string> listGUI = sofa::gui::GUIManager::ListSupportedGUI();
    //Insert default GUI
    {
        QAction *act= new QAction(this, QString("default")+QString("Action"));
        act->setText( "default");
        act->setToggleAction( true ); act->setOn(true);
        act->addTo( runSofaGUI);
        listActionGUI.push_back(act);
        connect(act, SIGNAL( activated()), this, SLOT( GUIChanged() ));
    }
    //Add content of GUI Factory
    for (unsigned int i=0; i<listGUI.size(); ++i)
    {
        QAction *act= new QAction(this, QString(listGUI[i].c_str())+QString("Action"));
        act->setText( QString(listGUI[i].c_str()));
        act->setToggleAction( true );
        act->addTo( runSofaGUI);
        listActionGUI.push_back(act);
        connect(act, SIGNAL( activated()), this, SLOT( GUIChanged() ));
    }

    //----------------------------------------------------------------------
    //Add menu Preset
    preset = new Q3PopupMenu(this);

    //----------------------------------------------------------------------
    //Add menu Window: to quickly find an opened simulation
    windowMenu = new Q3PopupMenu(this);
    this->menubar->insertItem(tr(QString("&Scenes")), windowMenu, menuIndex++);

    connect(windowMenu, SIGNAL(activated(int)), this, SLOT( changeCurrentScene(int)));



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

                presetArchitecture.insert(std::make_pair( directory, std::make_pair( namePreset, nameFile) ) );
            }

        }
    }
    end.close();

    std::multimap< std::string, std::pair< std::string,std::string> >::iterator it_preset = presetArchitecture.begin();
    while(it_preset != presetArchitecture.end())
    {
        std::string directoryName = it_preset->first;
        Q3PopupMenu* directory = new Q3PopupMenu(this);
        connect( directory, SIGNAL(activated(int)), this, SLOT(loadPreset(int)));
        preset->insertItem(QIconSet(), tr( it_preset->first.c_str()), directory);
        for (unsigned int i=0; i<presetArchitecture.count(directoryName); i++,it_preset++)
        {
            directory->insertItem(it_preset->second.first.c_str());//, this, SLOT(loadPreset()) );
            mapPreset.insert(it_preset->second);
        }
    }
    //----------------------------------------------------------------------
    //Recently Opened Files: if the init file does not exist, we create one
    std::string scenes ( "config/Modeler.ini" );
    if ( !sofa::helper::system::DataRepository.findFile ( scenes ) )
    {
        std::string fileToBeCreated = sofa::helper::system::DataRepository.getFirstPath() + "/" + scenes;

        std::ofstream ofile(fileToBeCreated.c_str());
        ofile << "";
        ofile.close();
    }

    connect( recentlyOpened, SIGNAL(activated(int)), this, SLOT(fileRecentlyOpened(int)));
    updateRecentlyOpened("");


    //----------------------------------------------------------------------
    //Center the application in the screen
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2,  ( screen.height() - this->height()) / 2  );


    connect( this->infoItem, SIGNAL(linkClicked( const QString &)), this, SLOT(fileOpen(const QString &)));
};



void SofaModeler::rebuildLibrary()
{
    library->clear();
    library->build(exampleFiles);
}

void SofaModeler::closeEvent( QCloseEvent *e)
{
    const int numTab=sceneTab->count();
    for (int i=0; i<numTab; ++i) closeTab();

    e->accept();
}

void SofaModeler::fileNew( GNode* root)
{
    if (!root) graph->setFilename("");
    changeNameWindow("");

    //no parent, adding root: if root is NULL, then an empty GNode will be created
    root = graph->setRoot( root, false);
    sceneTab->setCurrentPage( sceneTab->count()-1);
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
}

void SofaModeler::createTab()
{
    QWidget *newtab = new QWidget();
    tabGraph = newtab;
    QVBoxLayout *currentTabLayout = new QVBoxLayout(newtab, 0,1, QString("ModelerScene"));
    sceneTab->addTab(newtab, QString("New Scene"));
    GraphModeler* modelerGraph = new GraphModeler(newtab,"Modeler");
    mapGraph.insert(std::make_pair(newtab, modelerGraph));
    mapGraph[newtab] = modelerGraph;
    graph = modelerGraph;

    graph->setAcceptDrops(true);
    currentTabLayout->addWidget(graph,0,0);

    graph->setSofaLibrary(library);
    graph->setPreset(preset);
    fileNew();

#ifdef SOFA_QT4
    connect(graph, SIGNAL(currentChanged(Q3ListViewItem *)), this, SLOT(changeInformation(Q3ListViewItem *)));
#else
    connect(graph, SIGNAL(currentChanged(QListViewItem *)), this, SLOT(changeInformation(QListViewItem *)));
#endif
    connect(graph, SIGNAL( fileOpen(const QString&)), this, SLOT(fileOpen(const QString&)));
    connect(graph, SIGNAL( undo(bool)), this, SLOT(updateUndo(bool)));
    connect(graph, SIGNAL( redo(bool)), this, SLOT(updateRedo(bool)));
}

void SofaModeler::closeTab()
{
    if (sceneTab->count() == 0) return;
    closeTab(tabGraph);
}

void SofaModeler::closeTab(int i)
{
    if (i<0) return;
    closeTab(sceneTab->page(i));
}

void SofaModeler::closeTab(QWidget *curTab)
{
    //If the scene has been launch in Sofa
    if (mapSofa.size() &&
        mapSofa.find(curTab) != mapSofa.end())
    {
        typedef std::multimap< const QWidget*, Q3Process* >::iterator multimapIterator;
        std::pair< multimapIterator,multimapIterator > range;
        range=mapSofa.equal_range(curTab);
        for (multimapIterator it=range.first; it!=range.second; it++)
        {
            removeTemporaryFiles(it->second->name());
            it->second->kill();
        }
        mapSofa.erase(range.first, range.second);
    }

    //Find the scene in the window menu
    std::map< int, QWidget* >::const_iterator it;
    for (it = mapWindow.begin(); it!=mapWindow.end(); it++)
    {
        if (it->second == curTab) break;
    }
    windowMenu->removeItem(it->first);
    mapWindow.erase(it->first);

    //Closing the Modify Dialog opened
    GraphModeler *mod = mapGraph[curTab];
    if (dynamic_cast< GraphModeler* >(mod))
    {
        mod->closeDialogs();
        mod->close();
    }


    sceneTab->removePage(curTab);
    mapGraph.erase(curTab);
    curTab->close();
}

void SofaModeler::fileOpen(std::string filename)
{
    if ( sofa::helper::system::DataRepository.findFile ( filename ) )
    {
        filename =  sofa::helper::system::DataRepository.getFile ( filename );
        openPath = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
        GNode *root = NULL;
        xml::BaseElement* newXML=NULL;
        if (!filename.empty())
        {
            sofa::helper::system::SetDirectory chdir ( filename );
            newXML = xml::loadFromFile ( filename.c_str() );
            if (newXML == NULL) return;
            if (!newXML->init()) std::cerr<< "Objects initialization failed.\n";
            root = dynamic_cast<GNode*> ( newXML->getObject() );
        }
        if (root)
        {
            createTab();
            fileNew(root);
            sceneTab->setCurrentPage( sceneTab->count()-1);

            graph->setFilename(filename);
            sceneTab->setTabLabel(tabGraph, QString(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).c_str()));
            sceneTab->setTabToolTip(tabGraph, QString(filename.c_str()));

            changeNameWindow(graph->getFilename());

            mapWindow.insert(std::make_pair(windowMenu->insertItem( graph->getFilename().c_str()), tabGraph));
        }
    }
}

void SofaModeler::fileRecentlyOpened(int id)
{
    fileOpen(recentlyOpened->text(id));
}

void SofaModeler::updateRecentlyOpened(std::string fileLoaded)
{

#ifdef WIN32
    for (unsigned int i=0; i<fileLoaded.size(); ++i)
    {
        if (fileLoaded[i] == '\\') fileLoaded[i] = '/';
    }
#endif
    std::string scenes ( "config/Modeler.ini" );

    scenes = sofa::helper::system::DataRepository.getFile ( scenes );

    std::vector< std::string > list_files;
    std::ifstream end(scenes.c_str());
    std::string s;
    while( std::getline(end,s) )
    {
        if (strcmp(s.c_str(),fileLoaded.c_str()))
            list_files.push_back(sofa::helper::system::DataRepository.getFile(s));
    }
    end.close();


    recentlyOpened->clear();

    std::ofstream out;
    out.open(scenes.c_str(),std::ios::out);
    if (sofa::helper::system::DataRepository.findFile(fileLoaded))
    {
        fileLoaded = sofa::helper::system::DataRepository.getFile(fileLoaded);
        out << fileLoaded << "\n";

        recentlyOpened->insertItem(QString(fileLoaded.c_str()));
    }
    for (unsigned int i=0; i<list_files.size() && i<MAX_RECENTLY_OPENED; ++i)
    {
        recentlyOpened->insertItem(QString(list_files[i].c_str()));
        out << list_files[i] << "\n";
    }

    out.close();
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
    else 	                          fileSave(graph->getFilename());
}

void SofaModeler::fileSave(std::string filename)
{
    simulation::tree::getSimulation()->exportXML(graph->getRoot(), filename.c_str(), true);
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

        std::string extension=sofa::helper::system::SetDirectory::GetExtension(s.ascii());
        if (extension.empty()) s+=QString(".scn");

        fileSave ( s.ascii() );
//  	    if (graph->getFilename().empty())
//  	      {
        std::string filename = s.ascii();
        graph->setFilename(filename);
        changeNameWindow(filename);
        sceneTab->setTabLabel(tabGraph, QString(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).c_str()));
        sceneTab->setTabToolTip(tabGraph, QString(filename.c_str()));
        examplePath = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
//  	      }

    }
}

void SofaModeler::loadPreset(int id)
{
    Q3PopupMenu *s = (Q3PopupMenu*) sender();
    std::string presetFile = presetPath+ mapPreset[s->text(id).ascii()];


    if (sofa::helper::system::DataRepository.findFile ( presetFile ))
    {
        presetFile = sofa::helper::system::DataRepository.getFile ( presetFile );
        graph->loadPreset(presetFile);
    }
    else std::cerr<<"Preset : " << presetFile << " Not found\n";
}

void SofaModeler::changeInformation(Q3ListViewItem *item)
{
    if (!item) return;
    if (item->childCount() != 0) return;
    std::string nameObject = item->text(0).ascii();
    std::string::size_type end_name = nameObject.find(" ");
    if (end_name != std::string::npos) nameObject.resize(end_name);
    changeComponent( library->getComponentDescription(nameObject) );
}


void SofaModeler::componentDraggedReception( std::string description, std::string // categoryName
        , std::string templateName, ClassEntry* componentEntry)
{
    changeComponent(description );
    if (!graph) return;
    graph->setLastSelectedComponent(templateName, componentEntry);
    Q3TextDrag *dragging = new Q3TextDrag(QString("ComponentCreation"), (QWidget*) this->sender());
    dragging->setText(QString("ComponentCreation"));
    dragging->dragCopy();
}

void SofaModeler::changeComponent(const std::string &description)
{
    infoItem->setText(description.c_str());
}


void SofaModeler::newGNode()
{
    Q3TextDrag *dragging = new Q3TextDrag(QString("GNode"), (QPushButton*)sender());
    dragging->setText(QString("GNode"));
    dragging->dragCopy();
}


void SofaModeler::changeCurrentScene( QWidget* currentGraph)
{
    tabGraph=currentGraph;
    graph = mapGraph[currentGraph];
    if (graph)
    {
        changeNameWindow(graph->getFilename());
        editUndoAction->setEnabled(graph->isUndoEnabled());
        editRedoAction->setEnabled(graph->isRedoEnabled());
    }
    else
    {
        editUndoAction->setEnabled(false);
        editRedoAction->setEnabled(false);
    }
}


void SofaModeler::changeCurrentScene(int id)
{
    sceneTab->setCurrentPage( sceneTab->indexOf(mapWindow[id]) );
}


void SofaModeler::changeNameWindow(std::string filename)
{

    std::string str = "Sofa Modeler";
    if (!filename.empty()) str+= std::string(" - ") + filename;
#ifdef WIN32
    setWindowTitle ( str.c_str() );
#else
    setCaption ( str.c_str() );
#endif
    updateRecentlyOpened(filename);
}

void SofaModeler::dropEvent(QDropEvent* event)
{
    QPushButton *push = (QPushButton *)event->source();
    if (push) push->setDown(false);

    QString text;
    Q3TextDrag::decode(event, text);
    std::string filename(text.ascii());
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

void SofaModeler::runInSofa()
{
    if (sceneTab->count() == 0) return;
    GNode* root=graph->getRoot();
    if (!root) return;
    // Init the scene
    sofa::gui::GUIManager::Init("Modeler");

    //Saving the scene in a temporary file ==> doesn't modify the current GNode of the simulation
    std::string path;
    if (graph->getFilename().empty()) path=presetPath;
    else path = sofa::helper::system::SetDirectory::GetParentDir(graph->getFilename().c_str())+std::string("/");

    std::string filename=path + std::string("temp") + (count++) + std::string(".scn");
    simulation::tree::getSimulation()->exportXML(root,filename.c_str(),true);


    if (count > '9') count = '0';

    QString messageLaunch;
    QStringList argv;
    //=======================================
    // Run Sofa
    if (sofaBinary.empty())
    {
#ifdef WIN32
        sofaBinary = binPath + "runSofa.exe";
#else
        sofaBinary = binPath + "runSofa";
#endif
    }
#ifdef WIN32
    argv << "runSofa.exe";
#else
    argv << "runSofa";
#endif

    argv << QString(filename.c_str());
    messageLaunch = QString("Use command: ")
            + QString(sofaBinary.c_str())
            + QString(" ");

    //Setting the GUI
    for (unsigned int i=0; i<listActionGUI.size(); ++i)
    {
        if (listActionGUI[i]->isOn())
        {
            if (std::string(listActionGUI[i]->text().ascii()) != "default")
            {
                argv << "-g" << listActionGUI[i]->text();
                messageLaunch += QString("-g ") + QString(listActionGUI[i]->text());
            }
            break;
        }
    }

    argv << "-t";


    Q3Process *p = new Q3Process(argv, this);
    p->setName(filename.c_str());
    p->setWorkingDirectory( QDir(binPath.c_str()) );
    connect(p, SIGNAL(processExited()), this, SLOT(sofaExited()));
    QDir dir(QString(sofa::helper::system::SetDirectory::GetParentDir(graph->getFilename().c_str()).c_str()));
    connect(p, SIGNAL( readyReadStdout () ), this , SLOT ( redirectStdout() ) );
    connect(p, SIGNAL( readyReadStderr () ), this , SLOT ( redirectStderr() ) );
    p->start();
    mapSofa.insert(std::make_pair(tabGraph, p));

    statusBar()->message(messageLaunch,5000);
}

void SofaModeler::redirectStdout()
{
    Q3Process* p = ((Q3Process*) sender());
    if( !p )
    {
        return;
    }
    QString data;
    while(p->canReadLineStdout())
    {
        data = p->readLineStdout();
        std::cout << data.ascii() << std::endl;
    }

}

void SofaModeler::redirectStderr()
{
    Q3Process* p = ((Q3Process*) sender());
    if( !p )
    {
        return;
    }
    QString data;
    while(p->canReadLineStderr())
    {
        data = p->readLineStderr();
        std::cout << data.ascii() << std::endl;
    }
}

void SofaModeler::sofaExited()
{
    Q3Process *p = ((Q3Process*) sender());
    removeTemporaryFiles(p->name());
    if (p->normalExit()) return;
    typedef std::multimap< const QWidget*, Q3Process* >::iterator multimapIterator;
    for (multimapIterator it=mapSofa.begin(); it!=mapSofa.end(); it++)
    {
        if (it->second == p)
        {
            const QString caption("Problem");
            const QString warning("Error running Sofa");
            QMessageBox::critical( this, caption,warning, QMessageBox::Ok | QMessageBox::Escape, QMessageBox::NoButton );
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
    filename += "." + std::string(sofa::gui::GUIManager::GetValidGUIName()) + ".view";
    //Remove eventual .view file
    ::remove(filename.c_str());
    //Remove eventual copy buffer
    ::remove(copyBuffer.c_str());
}


/// When the user enter the Modeler, grabbing something: determine the acceptance or not
void SofaModeler::dragEnterEvent( QDragEnterEvent* event)
{
    QString text;
    Q3TextDrag::decode(event, text);
    if (text.isEmpty()) event->ignore();
    else
    {
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file")  event->accept();
        else event->ignore();
    }
}

/// When the user move the mouse around, with something grabbed
void SofaModeler::dragMoveEvent( QDragMoveEvent* event)
{
    QString text;
    Q3TextDrag::decode(event, text);
    if (text.isEmpty()) event->ignore();
    else
    {
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file")  event->accept();
        else event->ignore();
    }
}


/// Quick Filter of the components
void SofaModeler::searchText(const FilterQuery& query)
{
#ifdef SOFA_QT4
    QSofaTreeLibrary* l=static_cast<QSofaTreeLibrary*>(library);
#else
    QSofaLibrary* l=static_cast<QSofaLibrary*>(library);
#endif
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
        sofaBinary=s.ascii();
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
        listActionGUI[i]->setOn(listActionGUI[i] == act);
    }
}

/*****************************************************************************************************************/
//Cut/Copy Paste management
void SofaModeler::editCut()
{
    if (graph)
    {
        isPasteReady=graph->editCut(presetPath+"copyBuffer.scn");
        editPasteAction->setEnabled(isPasteReady);
    }
}
void SofaModeler::editCopy()
{
    if (graph)
    {
        isPasteReady=graph->editCopy(presetPath+"copyBuffer.scn");
        editPasteAction->setEnabled(isPasteReady);
    }
}
void SofaModeler::editPaste()
{
    if (graph)
    {
        graph->editPaste(presetPath+"copyBuffer.scn");
    }
}

void SofaModeler::showPluginManager()
{
    SofaPluginManager::getInstance()->show();
}



}
}
}
