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
#include "GraphVisitor.h"
#include "WindowVisitor.h"


#include <tinyxml.cpp>
#include <tinyxmlerror.cpp>
#include <tinystr.cpp>
#include <tinyxmlparser.cpp>

#ifndef SOFA_QT4
typedef QListViewItem Q3ListViewItem;
#endif
namespace sofa
{

namespace gui
{

namespace qt
{

bool GraphVisitor::load(std::string &file)
{
    //Open it using TinyXML
    TiXmlDocument doc(file.c_str());
    bool loadOk = doc.LoadFile();
    if (!loadOk) return false;

    TiXmlHandle hDoc(&doc);
    TiXmlNode* pElem;
    //Getting the root of the file
    pElem=hDoc.FirstChildElement().Element();

    // should always have a valid root but handle gracefully if it does
    if (!pElem) return false;

    openNode( pElem, NULL, NULL);
    return true;
}


void GraphVisitor::openAttribute      ( TiXmlElement* element, Q3ListViewItem* item)
{
    if (!element) return;
    TiXmlAttribute* attribute=element->FirstAttribute();
    while (attribute)
    {
        std::string nameOfAttribute(attribute->Name());
        std::string valueOfAttribute(attribute->Value());

        addInformation(item, nameOfAttribute, valueOfAttribute);
        attribute=attribute->Next();
    }

}
Q3ListViewItem* GraphVisitor::openNode( TiXmlNode* node, Q3ListViewItem* parent, Q3ListViewItem* elementAbove)
{
    if (!node) return NULL;

    std::string nameOfNode=node->Value();
    int typeOfNode=node->Type();
    Q3ListViewItem *graphNode=NULL;
    switch (typeOfNode)
    {
    case TiXmlNode::DOCUMENT:
        break;

    case TiXmlNode::ELEMENT:
        graphNode = addNode(parent, elementAbove, nameOfNode);
        openAttribute( node->ToElement(), graphNode);
        break;

    case TiXmlNode::COMMENT:
        graphNode = addComment(parent, elementAbove, nameOfNode);
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

    Q3ListViewItem *element=NULL;
    for ( TiXmlNode* child = node->FirstChild(); child != 0; child = child->NextSibling())
    {
        element = openNode( child, graphNode, element);
    }
    return graphNode;
}


Q3ListViewItem *GraphVisitor::addNode(Q3ListViewItem *parent, Q3ListViewItem *elementAbove, std::string name)
{
    Q3ListViewItem *item=NULL;
    if (!parent)
    {
        //Add a Root
        item=new Q3ListViewItem(graph, QString(name.c_str()));
        item->setOpen(true);
    }
    else
    {
        //Add a child to a node
        item=new Q3ListViewItem(parent,elementAbove, QString(name.c_str()));
    }
    if (name == "Node")
        item->setPixmap(0,*WindowVisitor::getPixmap(WindowVisitor::NODE));
    else if (name == "Component")
        item->setPixmap(0,*WindowVisitor::getPixmap(WindowVisitor::COMPONENT));
    else if (name != "Vector")
        item->setPixmap(0,*WindowVisitor::getPixmap(WindowVisitor::OTHER));
    item->setMultiLinesEnabled(true);
    return item;
}

void GraphVisitor::addInformation(Q3ListViewItem *element, std::string name, std::string info)
{
    if (!element) return;
    if (element->text(0) == QString("Node"))
        element->setText(0, QString(info.c_str()));
    else  if (element->text(0) == QString("Component"))
        element->setText(0, QString(info.c_str()));
    else
    {
        if (element->text(1).isEmpty())
        {
            element->setText(1, QString( name.c_str()));
            element->setText(2, QString( info.c_str()));
        }
        else
        {
            QString nameQt = element->text(1) + QString("\n") + QString( name.c_str());
            QString infoQt = element->text(2) + QString("\n") + QString( info.c_str());

            element->setText(1, nameQt);
            element->setText(2, infoQt);
        }
    }
}

Q3ListViewItem *GraphVisitor::addComment(Q3ListViewItem *element,Q3ListViewItem *elementAbove,  std::string comment)
{
    if (!element) return NULL;
    Q3ListViewItem *result = new Q3ListViewItem(element, elementAbove,QString(comment.c_str()));
    result->setPixmap(0,*WindowVisitor::getPixmap(WindowVisitor::COMMENT));
    result->setSelectable(false);
    result->setMultiLinesEnabled(true);
    return result;
}

}
}
}
