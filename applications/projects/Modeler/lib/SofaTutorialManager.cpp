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

#include "SofaTutorialManager.h"

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>

#ifdef SOFA_QT4
#include <QHBoxLayout>
#include <QUrl>
#include <QMenuBar>
#else
#include <qlayout.h>
#include <qurl.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

SofaTutorialManager::SofaTutorialManager(QWidget* parent, const char* name):QMainWindow(parent, name)
{

    QWidget *mainWidget = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(mainWidget);
    this->setCentralWidget(mainWidget);
    this->setAcceptDrops(TRUE);
    this->setCaption(QString("Sofa Tutorials"));

    std::string fileTutorials="config/tutorials.xml";
    fileTutorials = sofa::helper::system::DataRepository.getFile( fileTutorials );

    //Create the tree containing the tutorials
    selector = new TutorialSelector(fileTutorials, mainWidget);
    connect(selector, SIGNAL(openTutorial(const std::string&)),
            this, SLOT(openTutorial(const std::string&)));
    connect(selector, SIGNAL(openHTML(const std::string&)),
            this, SLOT(openHTML(const std::string&)));

    //Create the HTML Browser to display the information
    descriptionPage = new QTextBrowser(mainWidget);
#ifdef SOFA_QT4
    connect(descriptionPage, SIGNAL(anchorClicked(const QUrl&)), this, SLOT(dynamicChangeOfScene(const QUrl&)));
#else
    // QMimeSourceFactory::defaultFactory()->setExtensionType("html", "text/utf8");
    descriptionPage->mimeSourceFactory()->setExtensionType("html", "text/utf8");;
    connect(descriptionPage, SIGNAL(linkClicked(const QString&)), this, SLOT(dynamicChangeOfScene(const QString&)));
#endif
    descriptionPage->setMinimumWidth(400);
    //Create the Graph
    graph = new GraphModeler(mainWidget);
    graph->setAcceptDrops(true);

#ifdef SOFA_QT4
    //Create the Menu bar
    QMenuBar *menuBar = this->menuBar();
    menuBar->setEnabled(true);

    runInSofaAction = new QAction(this);
    runInSofaAction->setText("Launch scene in Sofa");
    runInSofaAction->setAccel(QKeySequence(tr("Ctrl+R")));
    menuBar->addAction(runInSofaAction);
    connect(runInSofaAction, SIGNAL(activated()), this, SLOT(launchScene()));

    QAction *undoAction = new QAction(this);
    undoAction->setAccel(QKeySequence(tr("Ctrl+Z")));
    menuBar->addAction(undoAction);
    connect(undoAction, SIGNAL(activated()), graph, SLOT(editUndo()));


    QAction *redoAction = new QAction(this);
    redoAction->setAccel(QKeySequence(tr("Ctrl+Y")));
    menuBar->addAction(redoAction);
    connect(redoAction, SIGNAL(activated()), graph, SLOT(editRedo()));
#else
    QToolBar *toolBar = new QToolBar( QString(""), this, DockTop);
    toolBar->setLabel(QString("Tools"));

    buttonRunInSofa = new QPushButton(QString("Launch scene in Sofa"), toolBar);
    connect(buttonRunInSofa, SIGNAL(clicked()), this, SLOT(launchScene()));

    runInSofaAction = new QAction(this);
    runInSofaAction->setText("Launch scene in Sofa");
    runInSofaAction->setAccel(QKeySequence(tr("Ctrl+R")));
    connect(runInSofaAction, SIGNAL(activated()), this, SLOT(launchScene()));

    QAction *undoAction = new QAction(this);
    undoAction->setAccel(QKeySequence(tr("Ctrl+Z")));
    connect(undoAction, SIGNAL(activated()), graph, SLOT(editUndo()));


    QAction *redoAction = new QAction(this);
    redoAction->setAccel(QKeySequence(tr("Ctrl+Y")));
    connect(redoAction, SIGNAL(activated()), graph, SLOT(editRedo()));
#endif


    //Setup the layout
    mainLayout->addWidget(selector);
    mainLayout->addWidget(descriptionPage);
    mainLayout->addWidget(graph);

    this->resize(1000,600);
    this->setPaletteBackgroundColor(QColor(255,180,120));
    openHTML("");


    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/SOFATUTORIALS.png" )).c_str();
#ifdef SOFA_QT4
    this->setWindowIcon(QIcon(pathIcon));
#else
    this->setIcon(QPixmap(pathIcon));
#endif
}

void SofaTutorialManager::openTutorial(const std::string& filename)
{
    if (filename.empty()) return;

    std::string file(sofa::helper::system::SetDirectory::GetFileName(filename.c_str()));
    runInSofaAction->setText(QString("Launch ")+QString(file.c_str()) + QString(" in Sofa"));
#ifndef SOFA_QT4
    buttonRunInSofa->setText(QString("Launch ")+QString(file.c_str()) + QString(" in Sofa"));
#endif

    //Set the Graph
    xml::BaseElement* newXML = xml::loadFromFile ( filename.c_str() );
    if (newXML == NULL) return;
    if (!newXML->init()) std::cerr<< "Objects initialization failed.\n";
    GNode *root = dynamic_cast<GNode*> ( newXML->getObject() );
    graph->setRoot(root, false);
    graph->setFilename(filename);
}

void SofaTutorialManager::openHTML(const std::string &filename)
{
    if (filename.empty())
    {
#ifdef WIN32
        static std::string defaultHTML("file:///"+sofa::helper::system::SetDirectory::GetProcessFullPath(sofa::helper::system::DataRepository.getFile( "Tutorials/Tutorials.html" ).c_str()));
#else
        static std::string defaultHTML(sofa::helper::system::SetDirectory::GetProcessFullPath(sofa::helper::system::DataRepository.getFile( "Tutorials/Tutorials.html" ).c_str()));
#endif
        openHTML(defaultHTML);
        return;
    }

#ifdef SOFA_QT4
#ifdef WIN32
    descriptionPage->setSource(QUrl(QString("file:///")+QString(filename.c_str())));
#else
    descriptionPage->setSource(QUrl(QString(filename.c_str())));
#endif
#else
    descriptionPage->mimeSourceFactory()->setFilePath(QString(filename.c_str()));
    descriptionPage->setSource(QString(filename.c_str()));
#endif
}

#ifdef SOFA_QT4
void SofaTutorialManager::dynamicChangeOfScene( const QUrl& u)
{
    std::string path=u.path().ascii();
#ifdef WIN32
    path = path.substr(1);
#endif
#else
void SofaTutorialManager::dynamicChangeOfScene( const QString& u)
{
    std::string path=u.ascii();
#endif
    path  = sofa::helper::system::DataRepository.getFile(path);
    std::string extension=sofa::helper::system::SetDirectory::GetExtension(path.c_str());
    if (extension == "xml" || extension == "scn")
    {
        openTutorial(path);
        std::string htmlLink=path.substr(0,path.size()-3)+"html";
        if (sofa::helper::system::DataRepository.findFile(htmlLink))
            openHTML(htmlLink);
        else
            openHTML("");
    }
    else if (extension == "html")
    {
        openHTML(path);
    }
}


void SofaTutorialManager::launchScene()
{
    emit runInSofa(graph->getFilename(), graph->getRoot());
}

}
}
}
