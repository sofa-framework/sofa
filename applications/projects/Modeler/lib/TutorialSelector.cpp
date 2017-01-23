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

#include "TutorialSelector.h"
#include "iconnode.xpm"

#include <iostream>

#include <sofa/helper/vector.h>

#include <QHeaderView>
#include <QImage>

namespace sofa
{

namespace gui
{

namespace qt
{


TutorialSelector::TutorialSelector(QWidget* parent):QTreeWidget(parent)
{
    connect (this, SIGNAL(itemDoubleClicked( QTreeWidgetItem *, int)),
            this, SLOT( changeRequested( QTreeWidgetItem *)));

    this->header()->hide();
    this->setSortingEnabled(false);

    //this->addColumn("Tutorials");
    //this->setColumnWidthMode(0,QTreeWidget::Maximum);

    itemToCategory.insert(std::make_pair((QTreeWidgetItem*)0, Category("All Sofa Tutorials", sofa::helper::system::DataRepository.getFile("Tutorials/Tutorials.xml"), sofa::helper::system::DataRepository.getFile("Tutorials/Tutorials.html"))));
}

void TutorialSelector::init()
{
    openCategory(QString("All Sofa Tutorials"));

    //Store all the categories we have for the software
    helper::vector< Category > allCategories;
    std::map< QTreeWidgetItem *, Category>::const_iterator itCategory;
    for (itCategory=itemToCategory.begin(); itCategory!=itemToCategory.end(); ++itCategory)
    {
        allCategories.push_back(itCategory->second);
    }

    //Associate a file loading a category to all its tutorials
    listTutoFromFile.clear();
    while (!allCategories.empty())
    {
        const Category& currentCategory=allCategories.back();

        openCategory(currentCategory);
        std::map< QTreeWidgetItem *, Tutorial>::const_iterator itTuto;
        for (itTuto=itemToTutorial.begin(); itTuto!=itemToTutorial.end(); ++itTuto)
        {
            listTutoFromFile.insert(std::make_pair(currentCategory, itTuto->second));
        }

        allCategories.pop_back();
    }
}

void TutorialSelector::loadTutorials(const std::string &fileTutorial)
{
    this->clear();
    itemToTutorial.clear();

    //Open it using TinyXML
    TiXmlDocument doc(fileTutorial.c_str());
    doc.LoadFile();

    TiXmlHandle hDoc(&doc);
    //Getting the root of the file
    TiXmlNode* node=hDoc.FirstChildElement().ToElement();
    if (!node)
    {
        std::cerr << "Error loading file: " << fileTutorial << std::endl;
        return;
    }
    openNode(node, 0, true);

    this->setMaximumWidth((int)(this->columnWidth(0)*1.1));
}

void TutorialSelector::openNode(TiXmlNode *node, QTreeWidgetItem *parent, bool isRoot)
{
    std::string nameOfNode=node->Value();
    // TinyXml API changed in 2.6.0, ELEMENT was replaced with TINYXML_ELEMENT
    // As the version number is not available as a macro, the most portable was is to
    // replace these constants with checks of the return value of ToElement(), ...
    // -- Jeremie A. 02/07/2011
    //int typeOfNode=node->Type();
    QTreeWidgetItem* item=0;
    if (node->ToElement())   // case TiXmlNode::ELEMENT:
    {
        if (!isRoot)
        {
            item = new QTreeWidgetItem();
            item->setText(0, QString(nameOfNode.c_str()));

            if (!parent)
            {
                this->addTopLevelItem(item);
                item->setExpanded(true);
            }
            else
            {
                parent->addChild(item);
                item->setExpanded(false);
            }
        }
        openAttribute(node->ToElement(), item);
    }
    else     // default:
    {
    }
    for ( TiXmlNode* child = node->FirstChild(); child != 0; child = child->NextSibling())
    {
        openNode(child, item);
    }
}

void TutorialSelector::openAttribute(TiXmlElement* element,  QTreeWidgetItem *item)
{
    if (!element || !item) return;
    TiXmlAttribute* attribute=element->FirstAttribute();
    std::string typeElement=element->Value() ;

    std::map<std::string, std::string> attributes;

    while (attribute)
    {
        const std::string &nameOfAttribute=(attribute->Name());
        const std::string &valueOfAttribute=(attribute->Value());

        attributes.insert(std::make_pair(nameOfAttribute,valueOfAttribute));
        attribute=attribute->Next();

        if (nameOfAttribute == "name")
        {
            item->setText(0, QString(valueOfAttribute.c_str()));
        }
    }

    if (typeElement == "Group")
    {
        static QPixmap pixNode((const char**)iconnode_xpm);
        item->setIcon(0, QIcon(pixNode));
    }
    else if (typeElement == "Category")
    {
        static QImage imageScene(QString(sofa::helper::system::DataRepository.getFile("textures/media-seek-forward.png").c_str()));
        static QPixmap pixScene;
        if (imageScene.width() != 20)
        {
            imageScene=imageScene.scaled(20,10, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            pixScene.convertFromImage(imageScene);
        }
        item->setIcon(0,QIcon(pixScene));

        Category C(item->text(0).toStdString(), attributes["xml"], attributes["html"]);
        if (C.htmlFilename.empty() && C.xmlFilename.size() >= 4)
        {
            std::string htmlFile=C.xmlFilename;
            //Open Description
            htmlFile = htmlFile.substr(0,htmlFile.size()-4);
            htmlFile += ".html";

            if ( sofa::helper::system::DataRepository.findFile (htmlFile) )
                htmlFile = sofa::helper::system::DataRepository.getFile ( htmlFile );
            else htmlFile.clear();

            C.htmlFilename=htmlFile;
        }
        if (!C.xmlFilename.empty())
        {
            if ( sofa::helper::system::DataRepository.findFile (C.xmlFilename) )
                C.xmlFilename = sofa::helper::system::DataRepository.getFile ( C.xmlFilename);
        }
        itemToCategory.insert(std::make_pair(item, C));
    }
    else if (typeElement == "Tutorial")
    {
        static QImage imageScene(QString(sofa::helper::system::DataRepository.getFile("icons/SOFA.png").c_str()));
        static QPixmap pixScene;
        if (imageScene.width() != 20)
        {
            imageScene=imageScene.scaled(20,20, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            pixScene.convertFromImage(imageScene);
        }

        item->setIcon(0,QIcon(pixScene));
        Tutorial T(item->text(0).toStdString(), attributes["scene"], attributes["html"]);
        if (T.htmlFilename.empty() && T.sceneFilename.size() >= 4)
        {
            std::string htmlFile=T.sceneFilename;
            //Open Description
            htmlFile = htmlFile.substr(0,htmlFile.size()-4);
            htmlFile += ".html";

            if ( sofa::helper::system::DataRepository.findFile (htmlFile) )
                htmlFile = sofa::helper::system::DataRepository.getFile ( htmlFile );
            else htmlFile.clear();

            T.htmlFilename=htmlFile;
        }

        if (!T.sceneFilename.empty())
        {
            if ( sofa::helper::system::DataRepository.findFile (T.sceneFilename) )
                T.sceneFilename = sofa::helper::system::DataRepository.getFile ( T.sceneFilename);
        }

        itemToTutorial.insert(std::make_pair(item, T));
    }
}

void TutorialSelector::changeRequested( QTreeWidgetItem *item )
{
    if (itemToTutorial.find(item) != itemToTutorial.end())
    {
        const Tutorial &T=itemToTutorial[item];
        openTutorial(T);
    }
    else if (itemToCategory.find(item) != itemToCategory.end())
    {
        const Category &C=itemToCategory[item];
        openCategory(C);
    }
}

void TutorialSelector::openCategory( const QString &name)
{
    std::map< QTreeWidgetItem *, Category>::const_iterator it;
    for (it=itemToCategory.begin(); it!=itemToCategory.end(); ++it)
    {
        if (it->second.name == name.toStdString())
        {
            openCategory(it->second);
            return;
        }
    }
}

void TutorialSelector::openTutorial( const Tutorial &T)
{
    emit openTutorial(T.sceneFilename);
    emit openHTML(T.htmlFilename);
}

void TutorialSelector::openCategory( const Category &C)
{
    emit openTutorial("");
    loadTutorials(C.xmlFilename);
    emit openCategory(C.name);
    emit openHTML(C.htmlFilename);
    currentCategory=C;
}

void TutorialSelector::usingScene(const std::string &filename)
{
    //Called by the exterior: a modification in the actual scene has been done.
    //We want to find if this file is from one tutorial of a specific category
    //If it is the case, we update the list
    this->clearSelection();
    std::multimap< Category, Tutorial >::const_iterator it;
    for (it=listTutoFromFile.begin(); it!=listTutoFromFile.end(); ++it)
    {
        const Category &c=it->first;
        const Tutorial &t=it->second;

        //If we open a category
        if (c.xmlFilename == filename)
        {
            if (currentCategory != c)
            {
                loadTutorials(c.xmlFilename); //update the list of tutorials corresponding
                currentCategory = c;
                emit openHTML(c.htmlFilename);
            }

            return;
        }
        else if (t.sceneFilename == filename) //If we open a tutorial
        {
            if (currentCategory != c)
            {
                loadTutorials(c.xmlFilename);
                currentCategory = c;
            }
            emit openHTML(t.htmlFilename);

            std::map< QTreeWidgetItem *, Tutorial>::const_iterator it;
            for (it=itemToTutorial.begin(); it!=itemToTutorial.end(); ++it)
            {
                //select in the list the current tutorial
                if (it->second.sceneFilename == filename)
                {
                    if (it->first) this->setCurrentItem(it->first);
                    break;
                }
            }

            return;
        }
    }
}

std::list< std::string >TutorialSelector::getCategories() const
{
    std::list<std::string> list;
    std::multimap< Category, Tutorial>::const_iterator it;
    std::string catName;
    for (it=listTutoFromFile.begin(); it!=listTutoFromFile.end(); ++it)
    {
        if (catName != it->first.name)
        {
            catName=it->first.name;
            list.push_back(catName);
        }
    }
    return list;
}

void TutorialSelector::keyPressEvent ( QKeyEvent * e )
{
    switch ( e->key() )
    {
    case Qt::Key_Return :
    case Qt::Key_Enter :
    {
        if (this->currentItem()) changeRequested(this->currentItem());
        break;
    }
    default:
    {
        QTreeWidget::keyPressEvent(e);
        break;
    }
    }
}

}
}
}
