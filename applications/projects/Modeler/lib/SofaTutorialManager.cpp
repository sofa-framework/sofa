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
#include <Q3ToolBar>
#else
#include <qlayout.h>
#include <qurl.h>
typedef QToolBar Q3ToolBar;
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

SofaTutorialManager::SofaTutorialManager(QWidget* parent, const char* name):Q3MainWindow(parent, name), tutorialList(0)
{

    QWidget *mainWidget = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(mainWidget);
    this->setCentralWidget(mainWidget);
    this->setAcceptDrops(TRUE);
    this->setCaption(QString("Sofa Tutorials"));


    //Create the tree containing the tutorials
    selector = new TutorialSelector(mainWidget);

    connect(selector, SIGNAL(openCategory(const std::string&)),
            this, SLOT(openCategory(const std::string&)));
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

    //Setup the layout
    mainLayout->addWidget(selector);
    mainLayout->addWidget(descriptionPage);
    mainLayout->addWidget(graph);

    connect(this, SIGNAL(undo()), graph, SIGNAL(undo()));
    connect(this, SIGNAL(redo()), graph, SIGNAL(redo()));

    //Creation of a Tool Bar
    Q3ToolBar *toolBar = new Q3ToolBar( this );
    toolBar->setLabel(QString("Tools"));

    //Add list of tutorials
    tutorialList = new QComboBox(toolBar);

    //Add button to launch a scene in Sofa
    buttonRunInSofa = new QPushButton(QString("Launch scene in Sofa"), toolBar);
    connect(buttonRunInSofa, SIGNAL(clicked()), this, SLOT(launchScene()));

    buttonEditInModeler = new QPushButton(QString("Edit in Modeler"), toolBar);
    connect(buttonEditInModeler, SIGNAL(clicked()), this, SLOT(editScene()));

    //Set up the list of tutorials
    selector->init();

    const std::list<std::string> &listTuto=selector->getCategories();
    for (std::list<std::string>::const_reverse_iterator it=listTuto.rbegin(); it!=listTuto.rend(); ++it)
    {
        tutorialList->insertItem(QString(it->c_str()));
    }
    connect(tutorialList, SIGNAL(activated(const QString&)), selector, SLOT(openCategory(const QString &)));

    //Select All Sofa Tutorials as selected set
    tutorialList->setCurrentItem(tutorialList->count()-1);


    this->resize(1000,600);
    this->setPaletteBackgroundColor(QColor(255,180,120));
    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/SOFATUTORIALS.png" )).c_str();
#ifdef SOFA_QT4
    this->setWindowIcon(QIcon(pathIcon));
#else
    this->setIcon(QPixmap(pathIcon));
#endif
}

void SofaTutorialManager::editScene()
{
    emit( editInModeler(graph->getFilename() ) );

}



void SofaTutorialManager::openTutorial(const std::string& filename)
{
    graph->closeDialogs();

    if (filename.empty())
    {
        graph->hide();
        buttonRunInSofa->hide();
        buttonEditInModeler->hide();
        return;
    }
    buttonRunInSofa->show();
    buttonEditInModeler->show();
    graph->show();
    std::string file=filename;
    const std::string &dirSofa = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
    std::string::size_type found=filename.find(dirSofa);
    if (found == 0) file = filename.substr(dirSofa.size()+1);

    buttonRunInSofa->setText(QString("Launch ")+QString(file.c_str()) + QString(" in Sofa"));

    //Set the Graph
    xml::BaseElement* newXML = xml::loadFromFile ( filename.c_str() );
    if (newXML == NULL) return;
    if (!newXML->init()) std::cerr<< "Objects initialization failed.\n";
    GNode *root = dynamic_cast<GNode*> ( newXML->getObject() );
    graph->setRoot(root, false);
    graph->setFilename(filename);
    selector->usingScene(filename);
}

void SofaTutorialManager::openHTML(const std::string &filename)
{
    if (filename.empty())
    {
        static std::string defaultHTML(sofa::helper::system::SetDirectory::GetProcessFullPath(sofa::helper::system::DataRepository.getFile( "Tutorials/Tutorials.html" ).c_str()));
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

void SofaTutorialManager::openCategory(const std::string&
#ifdef SOFA_QT4
        filename
#endif
                                      )
{

#ifdef SOFA_QT4
    if (tutorialList) tutorialList->setCurrentIndex(tutorialList->findText(QString(filename.c_str())));
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
    }
    else if (extension == "html")
    {
        openHTML(path);
    }
}

void SofaTutorialManager::keyPressEvent ( QKeyEvent * e )
{

    if (e->state() == Qt::ControlButton )
    {
        switch(e->key())
        {
        case Qt::Key_R:
        {
            //Pressed CTRL+R
            launchScene();
            return;
        }
        case Qt::Key_Y:
        {
            emit redo();
            return;
        }
        case Qt::Key_Z:
        {
            emit undo();
            return;
        }
        default: ;
        }
    }
    e->ignore();
}

void SofaTutorialManager::launchScene()
{
    emit runInSofa(graph->getFilename(), graph->getRoot());
}

}
}
}
