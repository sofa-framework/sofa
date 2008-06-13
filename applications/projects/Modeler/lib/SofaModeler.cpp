#include "SofaModeler.h"
#include <map>
#include <set>

#ifdef SOFA_QT4
#include <QToolBox>
#include <QSpacerItem>
#include <QGridLayout>
#include <QTextEdit>
#include <QComboBox>
#else
#include <qtoolbox.h>
#include <qlayout.h>
#include <qtextedit.h>
#include <qcombobox.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

#ifdef SOFA_QT4
typedef Q3TextDrag QTextDrag;
#endif


SofaModeler::SofaModeler()
{
    QWidget *GraphSupport = new QWidget((QWidget*)splitter2);
    QGridLayout* GraphLayout = new QGridLayout(GraphSupport, 1,1,5,2,"GraphLayout");
    graph = new GraphModeler(GraphSupport);
    graph->setAcceptDrops(true);
    GraphLayout->addWidget(graph,0,0);

    //Construction of the left part of the GUI: list of all objects sorted by base class
    std::set< std::string > setType;
    std::multimap< std::string, ClassInfo* > inventory;
    std::vector< ClassInfo* > entries;
    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries);
    for (unsigned int i=0; i<entries.size(); ++i)
    {
        sofa::core::objectmodel::BaseObject *object;
        if (entries[i]->creatorMap.find(entries[i]->defaultTemplate) != entries[i]->creatorMap.end())
        {
            object = entries[i]->creatorMap.find(entries[i]->defaultTemplate)->second->createInstance(NULL, NULL);
        }
        else
        {
            object = entries[i]->creatorList.begin()->second->createInstance(NULL, NULL);
        }

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
        delete object;
    }

    std::set< std::string >::iterator it;
    std::multimap< std::string,ClassInfo* >::iterator itMap;
    for (it = setType.begin(); it != setType.end(); it++)
    {
        itMap = inventory.find( (*it) );
        unsigned int numRows = inventory.count( (*it) );
        QWidget* gridWidget = new QWidget(SofaComponents, QString((*it)+"Widget"));
        QGridLayout* gridLayout = new QGridLayout( gridWidget, numRows+1,1);
        bool needSpacer = false;


        //Insert all the components belonging to the same family
        SofaComponents->addItem( gridWidget ,(*it) );
        for (unsigned int i=0; i< inventory.count( (*it) ); ++i)
        {
            ClassInfo* entry = itMap->second;
            QPushButton *button = new QPushButton(gridWidget, QString(entry->className));
            gridLayout->addWidget(button, i,0);
            button->setFlat(true);
            //Template: Add in a combo box the list of the templates
            QComboBox *combo=NULL;
            if (entry->creatorList.size() > 1)
            {
                needSpacer = true;
                combo = new QComboBox(gridWidget);
                gridLayout->addWidget(combo, i,1);
                std::list< std::pair< std::string, ClassCreator*> >::iterator it;
                for (it=entry->creatorList.begin(); it!= entry->creatorList.end(); it++)
                {
                    combo->insertItem(QString(it->first.c_str()));
                }
            }
            button->setText(QString(entry->className));

            mapComponents.insert(std::make_pair(button, std::make_pair(entry, combo)));
            itMap++;

            connect(button, SIGNAL(pressed() ), this, SLOT( dragComponent() ));
        }

        gridLayout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Minimum ), numRows,0);
        if (needSpacer) gridLayout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Minimum ), numRows,1);
    }


    graph->setLibrary(mapComponents);
    graph->fileNew();
    connect(graph, SIGNAL(changeNameWindow(std::string)), this, SLOT(changeNameWindow(std::string)));
    connect(graph, SIGNAL(currentChanged(QListViewItem *)), this, SLOT(changeInformation(QListViewItem *)));
};

void SofaModeler::changeInformation(QListViewItem *item)
{
    if (!item) return;
    if (item->childCount() != 0) return;
    std::string nameObject = item->text(0);
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

    QTextDrag *dragging = new QTextDrag(QString(currentComponent->className.c_str()), (QWidget *)sender);
    QComboBox *box = (QComboBox *) mapComponents.find(sender)->second.second;
    QString textDragged;
    if (box)  textDragged = box->currentText();

    dragging->setText(textDragged);
    dragging->dragCopy();

}


void SofaModeler::changeComponent(ClassInfo *currentComponent)
{
    std::string text;
    text  = std::string("<H2>")  + currentComponent->className + std::string(": ");
    for (std::set< std::string >::iterator it=currentComponent->baseClasses.begin(); it!=currentComponent->baseClasses.end() ; it++)
    {
        if (it != currentComponent->baseClasses.begin()) text += std::string(", ");
        text += (*it);
    }
    text += std::string("</H2>");

    text += std::string("<ul>");

    text += std::string("<li>Description: ") + currentComponent->description + std::string("</li>");

    text += std::string("<li>Authors: ")+currentComponent->authors;
    text += std::string("</li>");
    text += std::string("<li>License: ") + currentComponent->license + std::string("</li>");
    text += std::string("</ul>");

    infoItem->setText(text.c_str());

}


void SofaModeler::newGNode()
{
    QTextDrag *dragging = new QTextDrag(QString("GNode"), this);
    dragging->setText(QString("GNode"));
    dragging->dragCopy();
}

void SofaModeler::changeNameWindow(std::string filename)
{

    std::string str = "Sofa Modeler";
    if (!filename.empty()) str+= std::string(" - ") + filename;
#ifdef _WIN32
    setWindowTitle ( str.c_str() );
#else
    setCaption ( str.c_str() );
#endif
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
        fileExit();
        break;
    }
    default:
    {
        e->ignore();
        break;
    }
    }
}


}
}
}
