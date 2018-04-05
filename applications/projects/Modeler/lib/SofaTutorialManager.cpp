/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "SofaTutorialManager.h"

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/cast.h>

#include <SofaSimulationCommon/xml/XML.h>

#include <QHBoxLayout>
#include <QUrl>
#include <QToolBar>

namespace sofa
{

namespace gui
{

namespace qt
{

SofaTutorialManager::SofaTutorialManager(QWidget* parent, const char* name)
    :QMainWindow(parent), tutorialList(0)
{
    this->setObjectName(name);

    QWidget *mainWidget = new QWidget(this);
    QGridLayout *mainLayout = new QGridLayout(mainWidget);
    this->setCentralWidget(mainWidget);
    this->setAcceptDrops(true);
    this->setWindowTitle(QString("Sofa Tutorials"));

    //Add list of tutorials
    tutorialList = new QComboBox(mainWidget);
    tutorialList->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    //Add button to launch a scene in runSofa
    buttonRunInSofa = new QPushButton(QString("Launch scene in Sofa"), mainWidget);
    buttonRunInSofa->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
    connect(buttonRunInSofa, SIGNAL(clicked()), this, SLOT(launchScene()));

    //Add button to edit a scene in Modeler
    buttonEditInModeler = new QPushButton(QString("Edit in Modeler"), mainWidget);
    connect(buttonEditInModeler, SIGNAL(clicked()), this, SLOT(editScene()));

    //Create the tree containing the tutorials
    selector = new TutorialSelector(mainWidget);

    connect(selector, SIGNAL(openCategory(const std::string&)),
            this, SLOT(openCategory(const std::string&)));
    connect(selector, SIGNAL(openTutorial(const std::string&)),
            this, SLOT(openTutorial(const std::string&)));
    connect(selector, SIGNAL(openHTML(const std::string&)),
            this, SLOT(openHTML(const std::string&)));
    selector->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);


    //Create the HTML Browser to display the information
    descriptionPage = new QTextBrowser(mainWidget);
    connect(descriptionPage, SIGNAL(anchorClicked(const QUrl&)), this, SLOT(dynamicChangeOfScene(const QUrl&)));

    //Create the Graph
    graph = new GraphModeler(mainWidget);
    graph->setAcceptDrops(true);
    graph->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);

    //Setup the layout

    mainLayout->addWidget(tutorialList, 0, 0);
    mainLayout->addWidget(buttonRunInSofa, 0, 1);
    mainLayout->addWidget(buttonEditInModeler, 0, 2);
    mainLayout->addWidget(selector, 1, 0);
    mainLayout->addWidget(descriptionPage, 1, 1);
    mainLayout->addWidget(graph, 1, 2);

    //Set up the list of tutorials
    selector->init();

    const std::list<std::string> &listTuto=selector->getCategories();
    for (std::list<std::string>::const_reverse_iterator it=listTuto.rbegin(); it!=listTuto.rend(); ++it)
    {
        tutorialList->addItem(QString(it->c_str()));
    }
    connect(tutorialList, SIGNAL(activated(const QString&)), selector, SLOT(openCategory(const QString &)));

    //Select All Sofa Tutorials as selected set
    tutorialList->setCurrentIndex(tutorialList->count()-1);


    this->resize(1000,600);
    QPalette p ;
    p.setColor(QPalette::Base, QColor(255,180,120));
    this->setPalette(p);
    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/SOFATUTORIALS.png" )).c_str();
    this->setWindowIcon(QIcon(pathIcon));
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

    msg_info_when(!newXML->init(), "SofaTutorialManager")
            << "Objects initialization failed.";

    Node *root = down_cast<Node>( newXML->getObject()->toBaseNode() );
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

#ifdef WIN32
    descriptionPage->setSource(QUrl(QString("file:///")+QString(filename.c_str())));
#else
    descriptionPage->setSource(QUrl(QString(filename.c_str())));
#endif
}

void SofaTutorialManager::openCategory(const std::string& filename)
{
    if (tutorialList)
        tutorialList->setCurrentIndex(tutorialList->findText(QString(filename.c_str())));
}

void SofaTutorialManager::dynamicChangeOfScene( const QUrl& u)
{
    std::string path=u.path().toStdString();
#ifdef WIN32
    path = path.substr(1);
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

    if (e->modifiers() == Qt::ControlModifier )
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
