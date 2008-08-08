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
#include "SofaModeler.h"
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>




#include <map>
#include <set>
#include <cctype>


// #define TEST_CREATION_COMPONENT

#ifdef SOFA_QT4
#include <QToolBox>
#include <QSpacerItem>
#include <QGridLayout>
#include <QTextEdit>
#include <Q3TextBrowser>
#include <QComboBox>
#include <QLabel>
#include <QApplication>
#include <QMenuBar>
#else
#include <qtoolbox.h>
#include <qlayout.h>
#include <qtextedit.h>
#include <qtextbrowser.h>
#include <qcombobox.h>
#include <qapplication.h>
#include <qmenubar.h>
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
    QWidget *GraphSupport = new QWidget((QWidget*)splitter2);
    QGridLayout* GraphLayout = new QGridLayout(GraphSupport, 1,1,5,2,"GraphLayout");


#ifdef SOFA_QT4
    fileMenu->removeAction(Action);
#endif

    //Add menu Preset
    preset = new Q3PopupMenu(this);
    this->menubar->insertItem(QString("Preset"), preset, 4);


    examplePath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/examples/" );
    presetPath = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/applications/projects/Modeler/preset/" );
    std::string presetFile = presetPath + std::string("preset.ini" );

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
    std::set< std::string > setType;
    std::multimap< std::string, ClassInfo* > inventory;
    std::vector< ClassInfo* > entries;
    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries);
    for (unsigned int i=0; i<entries.size(); ++i)
    {
#ifdef	    TEST_CREATION_COMPONENT
        sofa::core::objectmodel::BaseObject *object;
        if (entries[i]->creatorMap.find(entries[i]->defaultTemplate) != entries[i]->creatorMap.end())
        {
            object = entries[i]->creatorMap.find(entries[i]->defaultTemplate)->second->createInstance(NULL, NULL);
        }
        else
        {
            object = entries[i]->creatorList.begin()->second->createInstance(NULL, NULL);
        }
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
            setType.insert("_Undefined_");
            inventory.insert(std::make_pair("_Undefined_", entries[i]));
        }

    }

    std::set< std::string >::iterator it;
    std::multimap< std::string,ClassInfo* >::iterator itMap;

    for (it = setType.begin(); it != setType.end(); it++)
    {
        itMap = inventory.find( (*it) );
        unsigned int numRows = inventory.count( (*it) );
        QString s=QString(it->c_str()) + QString("Widget");
        QWidget* gridWidget = new QWidget(SofaComponents, s);
        QGridLayout* gridLayout = new QGridLayout( gridWidget, numRows+1,1);
        gridLayout->addItem(new QSpacerItem(1,1,QSizePolicy::Minimum, QSizePolicy::Expanding ), 0,0);

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



            QPushButton *button = new QPushButton(gridWidget, QString(entry->className.c_str()));
            connect(button, SIGNAL(pressed()), this, SLOT( test()));
            gridLayout->addWidget(button, counterElem,0);
            button->setFlat(false);

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


            //Template: Add in a combo box the list of the templates
            QComboBox *combo=NULL;
            if ( entry->creatorList.size() > 1 )
            {
                combo = new QComboBox(gridWidget);

                for (unsigned int t=0; t<templateCombo.size(); ++t)
                    combo->insertItem(QString(templateCombo[t].c_str()));
                if (templateCombo.size() == 1) //Mapping with only one template possible
                {
                    combo->hide();
                    gridLayout->addWidget(new QLabel(QString(templateCombo[0].c_str()), gridWidget), counterElem, 1);
                }
                else
                {
                    gridLayout->addWidget(combo, counterElem,1);
                    if (templateCombo.size() == 0) {combo->hide(); button->hide(); counterElem--;}
                }

            }
            else
            {
                if (!entry->creatorList.begin()->first.empty())
                {
                    QLabel *templateDescription = new QLabel(QString(entry->creatorList.begin()->first.c_str()), gridWidget);
                    gridLayout->addWidget(templateDescription, counterElem,1);
                }
            }
            button->setText(QString(entry->className.c_str()));

            mapComponents.insert(std::make_pair(button, std::make_pair(entry, combo)));
            itMap++;

            connect(button, SIGNAL(pressed() ), this, SLOT( dragComponent() ));
            counterElem++;
        }
    }

    connect( this->infoItem, SIGNAL(linkClicked( const QString &)), this, SLOT(fileOpen(const QString &)));

    connect( recentlyOpened, SIGNAL(activated(int)), this, SLOT(fileRecentlyOpened(int)));

    sceneTab = new QTabWidget(GraphSupport);
    GraphLayout->addWidget(sceneTab,0,0);
    connect( sceneTab, SIGNAL(currentChanged( QWidget*)), this, SLOT( changeCurrentScene( QWidget*)));

    newTab();
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
};


void SofaModeler::fileNew( GNode* root)
{
    if (!root) graph->setFilename("");
    changeNameWindow("");

    GNode *current_root=graph->getRoot();
    if (current_root) graph->clearGraph();

    //no parent, adding root: if root is NULL, then an empty GNode will be created
    root = graph->addGNode(NULL, root);
    sceneTab->setCurrentPage( sceneTab->count()-1);
}

void SofaModeler::fileOpen()
{
    QString s = getOpenFileName ( this, QString(examplePath.c_str()),"Scenes (*.scn *.xml *.simu *.pscn)", "open file dialog",  "Choose a file to open" );
    if (s.length() >0)
    {
        fileOpen(s);
        examplePath = sofa::helper::system::SetDirectory::GetParentDir(s.ascii());
    }
}

void SofaModeler::clearTab()
{
    fileNew();
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
    tabGraph = new QWidget();
    QVBoxLayout *currentTabLayout = new QVBoxLayout(tabGraph, 0,1, QString("ModelerScene"));
    sceneTab->addTab(tabGraph, QString("New Scene"));

    graph = new GraphModeler(tabGraph);
    mapGraph.insert(std::make_pair(tabGraph, graph));

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
    connect(graph, SIGNAL( fileOpen(std::string)), this, SLOT(fileOpen(std::string)));
}

void SofaModeler::closeTab()
{

    if (sceneTab->count() <=1)
        fileNew();
    else if (tabGraph)
    {
        //If the scene has been launch in Sofa
        if (mapSofa[tabGraph] != 0)
        {
            mapSofa[tabGraph]->fileExit();
            //	      delete mapSofa [tabGraph];
            mapSofa .erase(tabGraph);
        }
        //else
        //	      delete mapGraph[tabGraph];
        mapGraph.erase(tabGraph);

        delete tabGraph;
    }
    graph->closeGraph();
}

void SofaModeler::fileOpen(std::string filename)
{
    createTab();
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
    fileNew(root);
    sceneTab->setCurrentPage( sceneTab->count()-1);

    graph->setFilename(filename);
    sceneTab->setTabLabel(tabGraph, QString(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).c_str()));
    sceneTab->setTabToolTip(tabGraph, QString(filename.c_str()));

    changeNameWindow(graph->getFilename());
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
    for (unsigned int i=0; i<list_files.size() && i<5; ++i)
    {
        recentlyOpened->insertItem(QString(list_files[i].c_str()));
        out << list_files[i] << "\n";
    }

    out.close();
}


void SofaModeler::fileSave()
{
    if (graph->getFilename().empty()) fileSaveAs();
    else 	                          fileSave(graph->getFilename());
}

void SofaModeler::fileSave(std::string filename)
{
    changeNameWindow(filename);
    getSimulation()->printXML(graph->getRoot(), filename.c_str());
}


void SofaModeler::fileSaveAs()
{
    QString s = sofa::gui::qt::getSaveFileName ( this, QString(examplePath.c_str()), "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
    {
        fileSave ( s.ascii() );

        std::string filename = s.ascii();
        sceneTab->setTabLabel(tabGraph, QString(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).c_str()));
        sceneTab->setTabToolTip(tabGraph, QString(filename.c_str()));
        examplePath = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
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



void SofaModeler::dragComponent()
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

    text += std::string("</H2>");

    text += std::string("<ul>");

    text += std::string("<li><b>Description: </b>") + currentComponent->description + std::string("</li>");
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
        changeNameWindow(graph->getFilename());
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

void SofaModeler::keyPressEvent ( QKeyEvent * e )
{
    // ignore if there are modifiers (i.e. CTRL of SHIFT)
#ifdef SOFA_QT4
    if (e->modifiers()) return;
#else
    if (e->state() & (Qt::KeyButtonMask)) return;
#endif
    switch ( e->key() )
    {
    case Qt::Key_Escape :
    case Qt::Key_Q :
    {
        if (isActiveWindow())  fileExit();
        break;
    }
    default:
    {
        e->ignore();
        break;
    }
    }
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
    GNode* root=graph->getRoot();
    if (!root) return;
    // Init the scene
    sofa::gui::SofaGUI::Init("Modeler");
    getSimulation()->init(root);
    //=======================================
    // Run the GUI
    std::string gui = sofa::gui::SofaGUI::GetGUIName();
    std::vector<std::string> plugins;

    if (sofa::gui::SofaGUI::Init("Modeler",gui.c_str())) return ;
    sofa::gui::qt::RealGUI *guiSofa = new sofa::gui::qt::RealGUI(gui.c_str());

    guiSofa->setScene(root);
    guiSofa->show();
    guiSofa->setFocus();
    mapSofa.insert(std::make_pair(tabGraph, guiSofa));
}


}
}
}
