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
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QTextDrag Q3TextDrag;
#endif


SofaModeler::SofaModeler()
{

    count='0';
    displayComponents=0;
    isPasteReady=false;
    editPasteAction->setEnabled(false);
    QWidget *GraphSupport = new QWidget((QWidget*)splitter2);
    QGridLayout* GraphLayout = new QGridLayout(GraphSupport, 1,1,5,2,"GraphLayout");

#ifdef SOFA_QT4
    fileMenu->removeAction(Action);

    //Temporary: desactivate with Qt4 the filter
    //LabelSearch->hide();
    //SearchEdit->hide();
#endif
    connect(GNodeButton, SIGNAL(pressed()), this, SLOT( releaseButton()));

    //----------------------------------------------------------------------
    //Add plugin manager window. ->load external libs
    pluginManager = new SofaPluginManager;
    pluginManager->hide();
    this->connect(pluginManager->buttonClose, SIGNAL(clicked() ),  this, SLOT( updateComponentList() ));

    int menuIndex=4;
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
    std::vector<std::string> listGUI = sofa::gui::SofaGUI::ListSupportedGUI();
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
    this->menubar->insertItem(tr(QString("&Preset")), preset, menuIndex++);

    //----------------------------------------------------------------------
    //Add menu Window: to quickly find an opened simulation
    windowMenu = new Q3PopupMenu(this);
    this->menubar->insertItem(tr(QString("&Scenes")), windowMenu, menuIndex++);

    connect(windowMenu, SIGNAL(activated(int)), this, SLOT( changeCurrentScene(int)));

    examplePath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/examples/" );
    binPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/bin/" );
    presetPath = examplePath + std::string("Objects/");
    std::string presetFile = std::string("config/preset.ini" );

    presetFile = sofa::helper::system::DataRepository.getFile ( presetFile );

    //store the kind and the name of the preset
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


    //Construction of the left part of the GUI: list of all objects sorted by base class
    updateComponentList();

    connect( this->infoItem, SIGNAL(linkClicked( const QString &)), this, SLOT(fileOpen(const QString &)));

    connect( recentlyOpened, SIGNAL(activated(int)), this, SLOT(fileRecentlyOpened(int)));

    sceneTab = new QTabWidget(GraphSupport);
    GraphLayout->addWidget(sceneTab,0,0);
    connect( sceneTab, SIGNAL(currentChanged( QWidget*)), this, SLOT( changeCurrentScene( QWidget*)));



    changeLibraryLabel(0);
    connect(SofaComponents, SIGNAL(currentChanged(int)), this, SLOT(changeLibraryLabel(int)));
    //Recently Opened Files
    std::string scenes ( "config/Modeler.ini" );
    if ( !sofa::helper::system::DataRepository.findFile ( scenes ) )
    {
        std::string fileToBeCreated = sofa::helper::system::DataRepository.getFirstPath() + "/" + scenes;

        std::ofstream ofile(fileToBeCreated.c_str());
        ofile << "";
        ofile.close();
    }

    scenes = sofa::helper::system::DataRepository.getFile ( scenes );

    updateRecentlyOpened("");

    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2,  ( screen.height() - this->height()) / 2  );

    GraphSupport->resize(200,550);

    Library->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
    SofaComponents->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
};




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
    QString s = getOpenFileName ( this, QString(examplePath.c_str()),"Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;All (*)", "open file dialog",  "Choose a file to open" );
    if (s.length() >0)
    {
        fileOpen(s);
        examplePath = sofa::helper::system::SetDirectory::GetParentDir(s.ascii());
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
        newScene = sofa::helper::system::DataRepository.getFile ( newScene);
        fileOpen(newScene);
        graph->setFilename("");
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

    graph->setLibrary(mapComponents);
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

    QWidget* curTab = tabGraph;
    //If the scene has been launch in Sofa
    if (mapSofa.size() &&
        mapSofa.find(curTab) != mapSofa.end())
    {
        typedef std::multimap< const QWidget*, Q3Process* >::iterator multimapIterator;
        std::pair< multimapIterator,multimapIterator > range;
        range=mapSofa.equal_range(curTab);
        for (multimapIterator it=range.first; it!=range.second; it++)
        {
            removeTemporaryFiles(it->second);
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
    GraphModeler *mod = graph;
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


void SofaModeler::fileSave()
{
    if (sceneTab->count() == 0) return;
    if (graph->getFilename().empty()) fileSaveAs();
    else 	                          fileSave(graph->getFilename());
}

void SofaModeler::fileSave(std::string filename)
{
    simulation::tree::getSimulation()->printXML(graph->getRoot(), filename.c_str(), true);
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
    ClassInfo *currentComponent = getInfoFromName(nameObject);
    if (currentComponent) changeComponent(currentComponent);
}

void SofaModeler::changeLibraryLabel(int index)
{
    Library->setItemLabel(0, QString("Sofa Components[") + QString::number(displayComponents) + QString("] : ")+SofaComponents->itemLabel(index));
}

void SofaModeler::newComponent()
{
    const QObject* sender = this->sender();
    //Change the frontal description of the object
    if ( mapComponents.find(sender) == mapComponents.end()) return;
    ClassInfo *currentComponent = mapComponents.find(sender)->second.first;
    changeComponent(currentComponent);

    Q3TextDrag *dragging = new Q3TextDrag(QString(currentComponent->className.c_str()), (QWidget *)sender);
    QComboBox *box = (QComboBox *) mapComponents.find(sender)->second.second;
    QString textDragged;
    if (box)
    {
        textDragged = box->currentText();
    }
    dragging->setText(textDragged);
    dragging->dragCopy();

}


void SofaModeler::changeComponent(ClassInfo *currentComponent)
{
    std::string text;
    text  = std::string("<H2>")  + currentComponent->className + std::string(": ");

    std::vector< std::string > possiblePaths;
    for (std::set< std::string >::iterator it=currentComponent->baseClasses.begin(); it!=currentComponent->baseClasses.end() ; it++)
    {
        if (it != currentComponent->baseClasses.begin()) text += std::string(", ");
        text += (*it);
        std::string baseClassName( *it );
        for (unsigned int i=0; i<baseClassName.size(); ++i)
        {
            if (isupper(baseClassName[i])) baseClassName[i] = tolower(baseClassName[i]);
        }
        if (baseClassName == "odesolver")            baseClassName="solver";
        if (baseClassName == "mastersolver")         baseClassName="solver";
        if (baseClassName == "topologicalmapping")   baseClassName="topology";
        if (baseClassName == "topologyobject")       baseClassName="topology";
        if (baseClassName == "collisionmodel")       baseClassName="collision";
        std::string path=std::string("Components/") + baseClassName + std::string("/") + currentComponent->className + std::string(".scn");


        if ( sofa::helper::system::DataRepository.findFile ( path ) )
            possiblePaths.push_back(sofa::helper::system::DataRepository.getFile ( path ));
    }
    if (possiblePaths.size() == 0)
    {
        std::string path=std::string("Components/misc/") + currentComponent->className + std::string(".scn");

        if ( sofa::helper::system::DataRepository.findFile ( path ) )
            possiblePaths.push_back(sofa::helper::system::DataRepository.getFile ( path ));
    }


    std::string nameSpace = sofa::core::objectmodel::Base::decodeNamespaceName(currentComponent->creatorList.begin()->second->type());


    text += std::string("</H2>");

    text += std::string("<ul>");

    text += std::string("<li><b>Description: </b>") + currentComponent->description + std::string("</li>");


    if (!nameSpace.empty())
        text += std::string("<li><b>NameSpace: </b>")+nameSpace +std::string("</li>");
    if (!currentComponent->authors.empty())
        text += std::string("<li><b>Authors: </b>")+currentComponent->authors +std::string("</li>");
    if (!currentComponent->license.empty())
        text += std::string("<li><b>License: </b>") + currentComponent->license + std::string("</li>");
    if (possiblePaths.size() != 0)
        text += std::string("<li><b>Example: </b><a href=\"")+possiblePaths[0]+std::string("\">") + possiblePaths[0] + std::string("</a></li>");

    text += std::string("</ul>");

    infoItem->setText(text.c_str());
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

ClassInfo* SofaModeler::getInfoFromName(std::string name)
{
    std::map< const QObject* , std::pair<ClassInfo*, QObject*> >::iterator it;
    for (it=mapComponents.begin(); it!= mapComponents.end(); it++)
    {
        if (it->second.first->className == name) return it->second.first;
    }
    return NULL;
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
    sofa::gui::SofaGUI::Init("Modeler");

    //Saving the scene in a temporary file ==> doesn't modify the current GNode of the simulation
    std::string path;
    if (graph->getFilename().empty()) path=presetPath;
    else path = sofa::helper::system::SetDirectory::GetParentDir(graph->getFilename().c_str())+std::string("/");

    std::string filename=path + std::string("temp") + (count++) + std::string(".scn");
    simulation::tree::getSimulation()->printXML(root,filename.c_str(),true);


    if (count > '9') count = '0';

    QString messageLaunch;
    //=======================================
    // Run Sofa
    if (sofaBinary.empty())
    {
// 	    changeSofaBinary();
// 	    if (sofaBinary.empty()) return; //No binary found

        //Set the default parameter: Sofa won't start if they are wrong
#ifdef WIN32
        sofaBinary = binPath + "runSofa.exe";
#else
        sofaBinary = binPath + "runSofa";
#endif
    }
    QStringList argv;
    argv << QString(sofaBinary.c_str()) << QString(filename.c_str());
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

    statusBar()->message(messageLaunch,5000);

    Q3Process *p = new Q3Process(argv, this);
    p->setName(filename.c_str());
    connect(p, SIGNAL(processExited()), this, SLOT(sofaExited()));
    QDir dir(QString(sofa::helper::system::SetDirectory::GetParentDir(graph->getFilename().c_str()).c_str()));
    p->setWorkingDirectory(dir);
    p->setCommunication(0);
    p->start();
    mapSofa.insert(std::make_pair(tabGraph, p));

    //Maybe switch to a multimap as several sofa can be launch from the same tab
}


void SofaModeler::sofaExited()
{
    Q3Process *p = ((Q3Process*) sender());
    removeTemporaryFiles(p);
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

void SofaModeler::removeTemporaryFiles(Q3Process *p)
{
    std::string filename(p->name());
    std::string copyBuffer(presetPath+"copyBuffer.scn");
    //Delete Temporary file
    ::remove(filename.c_str());
    filename += ".view";
    //Remove eventual .view file
    ::remove(filename.c_str());
    //Remove eventual copy buffer
    ::remove(copyBuffer.c_str());
}


void SofaModeler::releaseButton()
{
    QPushButton *push = (QPushButton *)sender();
    if (push) push->setDown(false);
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

/// Quick Filter of te components
void SofaModeler::searchText(const QString& text)
{
    unsigned int displayed=0;

    std::multimap< QWidget*, std::pair< QPushButton*, QComboBox*> >::iterator itMap;
    for (unsigned int p=0; p<pages.size(); ++p)
    {
        QWidget* page=pages[p].begin()->first;
        const unsigned int numComponents=pages[p].size();
        unsigned int counterHiddenComponents=0;
        for (itMap=pages[p].begin(); itMap!=pages[p].end(); itMap++)
        {
            QPushButton* button= itMap->second.first;
            QComboBox* combo= itMap->second.second;
            if (!button->text().contains(text,false))
            {
                counterHiddenComponents++;
                button->hide();
                if (combo) combo->hide();
            }
            else
            {
                displayed++;
                button->show();
                if (combo) combo->show();
            }
        }



        int idx=SofaComponents->indexOf(page);
        if (counterHiddenComponents == numComponents)
        {
            //Hide the page
            if (idx >= 0)
            {
#ifdef SOFA_QT4
                SofaComponents->removeItem(idx);
#else
                SofaComponents->removeItem(page);
#endif
            }
        }
        else
        {
            if (idx < 0)
            {
                SofaComponents->insertItem(p,page,page->name());
            }
        }
    }

    displayComponents = displayed;

    changeLibraryLabel(SofaComponents->currentIndex());
    SofaComponents->update();
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
    pluginManager->show();
}


void SofaModeler::updateComponentList()
{
    //clear the current component list
    for (int p=0; p<(int)pages.size(); ++p)
    {
        QWidget* page=pages[p].begin()->first;
        int idx=SofaComponents->indexOf(page);
        //Hide the page
        if (idx >= 0)
        {
#ifdef SOFA_QT4
            SofaComponents->removeItem(idx);
#else
            SofaComponents->removeItem(page);
#endif
        }
    }


    //create the new one
    std::set< std::string > setType;
    std::multimap< std::string, ClassInfo* > inventory;
    std::vector< ClassInfo* > entries;
    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries);

    for (unsigned int i=0; i<entries.size(); ++i)
    {
#ifdef	    TEST_CREATION_COMPONENT
        sofa::core::objectmodel::BaseObject *object;
        std::cout << "Creating " << entries[i]->className << "\n";
        if (entries[i]->creatorMap.find(entries[i]->defaultTemplate) != entries[i]->creatorMap.end())
        {
            object = entries[i]->creatorMap.find(entries[i]->defaultTemplate)->second->createInstance(NULL, NULL);
        }
        else
        {
            object = entries[i]->creatorList.begin()->second->createInstance(NULL, NULL);
        }
        std::cout << "Deleting " << entries[i]->className << "\n";
        delete object;
#endif
        std::set< std::string >::iterator it;
        for (it = entries[i]->baseClasses.begin(); it!= entries[i]->baseClasses.end(); it++)
        {
            setType.insert((*it));
            inventory.insert(std::make_pair((*it), entries[i]));
        }
        if (entries[i]->baseClasses.size() == 0)
        {
            setType.insert("_Miscellaneous");
            inventory.insert(std::make_pair("_Miscellaneous", entries[i]));
        }
    }


    std::set< std::string >::iterator it;
    std::multimap< std::string,ClassInfo* >::iterator itMap;

    for (it = setType.begin(); it != setType.end(); it++)
    {
        itMap = inventory.find( (*it) );
        unsigned int numRows = inventory.count( (*it) );
        QString s=QString(it->c_str());


        std::multimap< QWidget*, std::pair< QPushButton*, QComboBox*> > pageMap;
        QWidget* gridWidget = new QWidget(SofaComponents, s);

        QGridLayout* gridLayout = new QGridLayout( gridWidget, numRows+1,1);

        //Insert all the components belonging to the same family
        SofaComponents->addItem( gridWidget ,it->c_str() );
        unsigned int counterElem=1;
        for (unsigned int i=0; i< inventory.count( (*it) ); ++i)
        {
            ClassInfo* entry = itMap->second;
            //We must remove the mass from the ForceField list
            if ( *it == "ForceField")
            {
                std::set< std::string >::iterator baseClassIt;
                bool isMass=false;
                for (baseClassIt = entry->baseClasses.begin(); baseClassIt!= entry->baseClasses.end() && !isMass; baseClassIt++)
                {
                    isMass= (*baseClassIt == "Mass");
                }
                if (isMass) { itMap++; continue;}
            }

            //We must remove the topological container from the Topology
            if ( *it == "Topology")
            {
                std::set< std::string >::iterator baseClassIt;
                bool isMass=false;
                for (baseClassIt = entry->baseClasses.begin(); baseClassIt!= entry->baseClasses.end() && !isMass; baseClassIt++)
                {
                    isMass= (*baseClassIt == "TopologyObject");
                }
                if (isMass) { itMap++; continue;}
            }



            //Count the number of template usable: Mapping and MechanicalMapping must be separated
            std::vector< std::string > templateCombo;
            {
                std::list< std::pair< std::string, ClassCreator*> >::iterator itTemplate;
                for (itTemplate=entry->creatorList.begin(); itTemplate!= entry->creatorList.end(); itTemplate++)
                {
                    if (*it == "Mapping")
                    {
                        std::string mechanical = itTemplate->first;
                        mechanical.resize(10+7);
                        if (mechanical == "MechanicalMapping") continue;
                    }
                    else if ( *it == "MechanicalMapping")
                    {
                        std::string nonmechanical = itTemplate->first;
                        nonmechanical.resize(7);
                        if (nonmechanical == "Mapping") continue;
                    }
                    templateCombo.push_back(itTemplate->first);
                }
            }

            if (templateCombo.size() == 0 && entry->creatorList.size() > 1)
            { itMap++; continue;}

            displayComponents++;
            QPushButton *button = new QPushButton(gridWidget, QString(entry->className.c_str()));
            connect(button, SIGNAL(pressed()), this, SLOT( newComponent()));
            connect(button, SIGNAL(pressed()), this, SLOT( releaseButton()));
            gridLayout->addWidget(button, counterElem,0);
            button->setFlat(false);

            //Template: Add in a combo box the list of the templates
            QComboBox *combo=NULL;
            if ( entry->creatorList.size() > 1 )
            {
                combo = new QComboBox(gridWidget);

                for (unsigned int t=0; t<templateCombo.size(); ++t)
                    combo->insertItem(QString(templateCombo[t].c_str()));

                gridLayout->addWidget(combo, counterElem,1);
            }
            else
            {
                if (!entry->creatorList.begin()->first.empty())
                {
                    combo = new QComboBox(gridWidget);
                    combo->insertItem(QString(entry->creatorList.begin()->first.c_str()));
                    gridLayout->addWidget(combo, counterElem,1);
                }
            }

            button->setText(QString(entry->className.c_str()));

            mapComponents.insert(std::make_pair(button, std::make_pair(entry, combo)));
            pageMap.insert(std::make_pair(gridWidget,std::make_pair(button, combo)));
            itMap++;

            //connect(button, SIGNAL(pressed() ), this, SLOT( newComponent() ));
            counterElem++;
        }
        gridLayout->addItem(new QSpacerItem(1,1,QSizePolicy::Minimum, QSizePolicy::Expanding ), counterElem,0);

        pages.push_back(pageMap);
    }

    //update the graph library
    std::map<  const QWidget*, GraphModeler*>::iterator itgraph;
    for (itgraph = mapGraph.begin(); itgraph!=mapGraph.end() ; itgraph++)
        itgraph->second->setLibrary(mapComponents);
}

}
}
}
