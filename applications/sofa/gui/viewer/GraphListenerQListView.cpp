/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#ifdef QT_MODULE_QT3SUPPORT
#include <Q3PopupMenu.h>
#else
#include <qpopupmenu.h>
#endif

#include <GenGraphForm.h>
#include "GUIField.h"
#include "RealGUI.h"

namespace sofa
{

namespace gui
{

namespace guiviewer
{
#ifdef QT_MODULE_QT3SUPPORT
typedef Q3PopupMmenu QPopupMenu;
#else
typedef QPopupMenu Q3PopupMenu;
#endif

/*****************************************************************************************************************/
Q3ListViewItem* GraphListenerQListView::createItem(Q3ListViewItem* parent)
{
    Q3ListViewItem* last = parent->firstChild();
    if (last == NULL)
        return new Q3ListViewItem(parent);
    while (last->nextSibling()!=NULL)
        last = last->nextSibling();
    return new Q3ListViewItem(parent, last);
}


/*****************************************************************************************************************/
void GraphListenerQListView::addChild(GNode* parent, GNode* child)
{
    if (frozen) return;
    if (items.count(child))
    {
        Q3ListViewItem* item = items[child];
        if (item->listView() == NULL)
        {
            if (parent == NULL)
                widget->insertItem(item);
            else if (items.count(parent))
                items[parent]->insertItem(item);
            else
            {
                std::cerr << "Graph -> QT ERROR: Unknown parent node "<<parent->getName()<<std::endl;
                return;
            }
        }
    }
    else
    {
        Q3ListViewItem* item;
        if (parent == NULL)
            item = new Q3ListViewItem(widget);
        else if (items.count(parent))
            item = createItem(items[parent]);
        else
        {
            std::cerr << "Graph -> QT ERROR: Unknown parent node "<<parent->getName()<<std::endl;
            return;
        }
        if (std::string(child->getName(),0,7) != "default")
            item->setText(0, child->getName().c_str());
        QPixmap* pix = getPixmap(child);
        if (pix)
            item->setPixmap(0, *pix);
        item->setOpen(true);
        items[child] = item;
    }
    // Add all objects and grand-children
    MutationListener::addChild(parent, child);
}

/*****************************************************************************************************************/
void GraphListenerQListView::removeChild(GNode* parent, GNode* child)
{
    MutationListener::removeChild(parent, child);
    if (items.count(child))
    {
        delete items[child];
        items.erase(child);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::moveChild(GNode* previous, GNode* parent, GNode* child)
{
    if (frozen && items.count(child))
    {
        Q3ListViewItem* itemChild = items[child];
        if (items.count(previous)) //itemChild->listView() != NULL)
        {
            Q3ListViewItem* itemPrevious = items[previous];
            itemPrevious->takeItem(itemChild);
        }
        else
        {
            removeChild(previous, child);
        }
        return;
    }
    if (!items.count(child) || !items.count(previous))
    {
        addChild(parent, child);
    }
    else if (!items.count(parent))
    {
        removeChild(previous, child);
    }
    else
    {
        Q3ListViewItem* itemChild = items[child];
        Q3ListViewItem* itemPrevious = items[previous];
        Q3ListViewItem* itemParent = items[parent];
        itemPrevious->takeItem(itemChild);
        itemParent->insertItem(itemChild);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::addObject(GNode* parent, core::objectmodel::BaseObject* object)
{
    if (frozen) return;
    if (items.count(object))
    {
        Q3ListViewItem* item = items[object];
        if (item->listView() == NULL)
        {
            if (items.count(parent))
                items[parent]->insertItem(item);
            else
            {
                std::cerr << "Graph -> QT ERROR: Unknown parent node "<<parent->getName()<<std::endl;
                return;
            }
        }
    }
    else
    {
        Q3ListViewItem* item;
        if (items.count(parent))
            item = createItem(items[parent]);
        else
        {
            std::cerr << "Graph -> QT ERROR: Unknown parent node "<<parent->getName()<<std::endl;
            return;
        }
        std::string name = sofa::helper::gettypename(typeid(*object));
        std::string::size_type pos = name.find('<');
        if (pos != std::string::npos)
            name.erase(pos);
        if (std::string(object->getName(),0,7) != "default")
        {
            name += "  ";
            name += object->getName();
        }
        item->setText(0, name.c_str());
        QPixmap* pix = getPixmap(object);
        if (pix)
            item->setPixmap(0, *pix);
        items[object] = item;
    }
}


/*****************************************************************************************************************/
void GraphListenerQListView::removeObject(GNode* /*parent*/, core::objectmodel::BaseObject* object)
{
    if (items.count(object))
    {
        delete items[object];
        items.erase(object);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::moveObject(GNode* previous, GNode* parent, core::objectmodel::BaseObject* object)
{
    if (frozen && items.count(object))
    {
        Q3ListViewItem* itemObject = items[object];
        Q3ListViewItem* itemPrevious = items[previous];
        itemPrevious->takeItem(itemObject);
        return;
    }
    if (!items.count(object) || !items.count(previous))
    {
        addObject(parent, object);
    }
    else if (!items.count(parent))
    {
        removeObject(previous, object);
    }
    else
    {
        Q3ListViewItem* itemObject = items[object];
        Q3ListViewItem* itemPrevious = items[previous];
        Q3ListViewItem* itemParent = items[parent];
        itemPrevious->takeItem(itemObject);
        itemParent->insertItem(itemObject);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::freeze(GNode* groot)
{
    if (!items.count(groot)) return;
    frozen = true;
}
/*
void unfreeze(Q3ListViewItem* parent, GNode* node)
{
if (!items.count(node))
{
addChild(node->parent, node);
return;
}
Q3ListViewItem* item = items[node];
if (item->listView() == NULL)
{
if (parent)
parent->insertItem(item);
else
widget->insertItem(item);
}
for(GNode::ChildIterator it = groot->child.begin(), itend = groot->child.end(); it != itend; ++it)
{
unfreeze(item, *it);
}
for(GNode::ObjectIterator it = groot->object.begin(), itend = groot->object.end(); it != itend; ++it)
{
core::objectmodel::BaseObject* object = *it;
if (!items.count(object))
addObject(node, object);
else
{
Q3ListViewItem* itemObject = items[object];
if (itemObject->listView() == NULL)
item->insertItem(itemObject);
}
}
}
     */

/*****************************************************************************************************************/
void GraphListenerQListView::unfreeze(GNode* groot)
{
    if (!items.count(groot)) return;
    frozen = false;
    addChild(NULL, groot);
}




} //guiviewer
} //gui
} //sofa
