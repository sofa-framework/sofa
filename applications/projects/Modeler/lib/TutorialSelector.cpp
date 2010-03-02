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

#include "TutorialSelector.h"
#include "iconnode.xpm"

#include <tinyxml.cpp>
#include <tinyxmlerror.cpp>
#include <tinystr.cpp>
#include <tinyxmlparser.cpp>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>

#include <iostream>

#ifdef SOFA_QT4
#include <Q3Header>
#include <QImage>
#else
#include <qheader.h>
#include <qimage.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

TutorialSelector::TutorialSelector(const std::string &fileTutorial, QWidget* parent):Q3ListView(parent)
{
#ifdef SOFA_QT4
    connect (this, SIGNAL(doubleClicked( Q3ListViewItem *, const QPoint &, int )),
            this, SLOT( openTutorial( Q3ListViewItem *, const QPoint &, int )));
#else
    connect (this, SIGNAL(doubleClicked( QListViewItem *, const QPoint &, int )),
            this, SLOT( openTutorial( QListViewItem *, const QPoint &, int )));
#endif
    this->header()->hide();
    this->setSorting(-1);


    this->addColumn("Tutorials");
    this->setColumnWidthMode(0,Q3ListView::Maximum);

    init(fileTutorial);

    this->setMaximumWidth((int)(this->columnWidth(0)*1.1));
}

void TutorialSelector::init(const std::string &fileTutorial)
{
    //Open it using TinyXML
    TiXmlDocument doc(fileTutorial.c_str());
    doc.LoadFile();

    TiXmlHandle hDoc(&doc);
    //Getting the root of the file
    TiXmlNode* node=hDoc.FirstChildElement().Element();
    if (!node)
    {
        std::cerr << "Error loading file: " << fileTutorial << std::endl;
        return;
    }
    openNode(node, 0, true);

}

void TutorialSelector::openNode(TiXmlNode *node, Q3ListViewItem *parent, bool isRoot)
{
    std::string nameOfNode=node->Value();
    int typeOfNode=node->Type();
    Q3ListViewItem* item=0;

    switch (typeOfNode)
    {
    case TiXmlNode::DOCUMENT:
        break;

    case TiXmlNode::ELEMENT:

        if (!isRoot)
        {
            if (!parent)
            {
                Q3ListViewItem *last = this->firstChild();
                if (last == 0) item = new Q3ListViewItem(this, QString(nameOfNode.c_str()));
                else
                {
                    while (last->nextSibling() != 0) last = last->nextSibling();
                    item = new Q3ListViewItem(this, last, QString(nameOfNode.c_str()));
                }
                item->setOpen(true);
            }
            else
            {
                Q3ListViewItem *last = parent->firstChild();
                if (last == 0) item = new Q3ListViewItem(parent, QString(nameOfNode.c_str()));
                else
                {
                    while (last->nextSibling() != 0) last = last->nextSibling();
                    item = new Q3ListViewItem(parent, last, QString(nameOfNode.c_str()));
                }
                item->setOpen(false);
            }
        }

        openAttribute(node->ToElement(), item);
        break;

    case TiXmlNode::COMMENT:
        break;

    case TiXmlNode::UNKNOWN:
        break;

    case TiXmlNode::TEXT:
        break;

    case TiXmlNode::DECLARATION:
        break;
    default:
        break;

    }
    for ( TiXmlNode* child = node->FirstChild(); child != 0; child = child->NextSibling())
    {
        openNode(child, item);
    }
}

void TutorialSelector::openAttribute(TiXmlElement* element,  Q3ListViewItem *item)
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
        item->setPixmap(0, pixNode);
    }
    else if (typeElement == "Tutorial")
    {
        static QImage imageScene(QString(sofa::helper::system::DataRepository.getFirstPath().c_str()) + "/icons/SOFA.png");
        static QPixmap pixScene;
        if (imageScene.width() != 20)
        {
            imageScene=imageScene.smoothScale(20,20);
            pixScene.convertFromImage(imageScene);
        }

        item->setPixmap(0,pixScene);
        Tutorial T(item->text(0).ascii(), attributes["scene"], attributes["html"]);
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

void TutorialSelector::openTutorial( Q3ListViewItem *item, const QPoint &, int )
{
    if (itemToTutorial.find(item) != itemToTutorial.end())
    {
        const Tutorial &T=itemToTutorial[item];
        emit openTutorial(T.sceneFilename);
        emit openHTML(T.htmlFilename);
    }
}

}
}
}
