/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "GraphListenerQListView.h"
#include <sofa/simulation/common/Colors.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include "iconmultinode.xpm"
#include "iconnode.xpm"
#include "iconwarning.xpm"
#include "icondata.xpm"
#include "iconsleep.xpm"


namespace sofa
{
using namespace core::objectmodel;
namespace gui
{

namespace qt
{
//***********************************************************************************************************

static const int iconWidth=8;
static const int iconHeight=10;
static const int iconMargin=6;

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

QPixmap* getPixmap(core::objectmodel::Base* obj)
{
    using namespace sofa::simulation::Colors;
    unsigned int flags=0;

    if (dynamic_cast<core::objectmodel::BaseNode*>(obj))
    {
		if (dynamic_cast<core::objectmodel::BaseNode*>(obj)->getContext()->isSleeping())
		{
			static QPixmap pixNode((const char**)iconsleep_xpm);
			return &pixNode;
		}
		else
		{
			static QPixmap pixNode((const char**)iconnode_xpm);
			return &pixNode;
		}
        //flags |= 1 << NODE;
    }
    else if (dynamic_cast<core::objectmodel::BaseObject*>(obj))
    {
        if (dynamic_cast<core::objectmodel::ContextObject*>(obj))
            flags |= 1 << CONTEXT;
        if (dynamic_cast<core::BehaviorModel*>(obj))
            flags |= 1 << BMODEL;
        if (dynamic_cast<core::CollisionModel*>(obj))
            flags |= 1 << CMODEL;
        if (dynamic_cast<core::behavior::BaseMechanicalState*>(obj))
            flags |= 1 << MMODEL;
        if (dynamic_cast<core::behavior::BaseProjectiveConstraintSet*>(obj))
            flags |= 1 << PROJECTIVECONSTRAINTSET;
        if (dynamic_cast<core::behavior::BaseConstraintSet*>(obj))
            flags |= 1 << CONSTRAINTSET;
        if (dynamic_cast<core::behavior::BaseInteractionForceField*>(obj) &&
            dynamic_cast<core::behavior::BaseInteractionForceField*>(obj)->getMechModel1()!=dynamic_cast<core::behavior::BaseInteractionForceField*>(obj)->getMechModel2())
            flags |= 1 << IFFIELD;
        else if (dynamic_cast<core::behavior::BaseForceField*>(obj))
            flags |= 1 << FFIELD;
        if (dynamic_cast<core::behavior::BaseAnimationLoop*>(obj)
            || dynamic_cast<core::behavior::OdeSolver*>(obj))
            flags |= 1 << SOLVER;
        if (dynamic_cast<core::collision::Pipeline*>(obj)
            || dynamic_cast<core::collision::Intersection*>(obj)
            || dynamic_cast<core::collision::Detection*>(obj)
            || dynamic_cast<core::collision::ContactManager*>(obj)
            || dynamic_cast<core::collision::CollisionGroupManager*>(obj))
            flags |= 1 << COLLISION;
        if (dynamic_cast<core::BaseMapping*>(obj))
            flags |= 1 << ((dynamic_cast<core::BaseMapping*>(obj))->isMechanical()?MMAPPING:MAPPING);
        if (dynamic_cast<core::behavior::BaseMass*>(obj))
            flags |= 1 << MASS;
        if (dynamic_cast<core::topology::Topology *>(obj)
            || dynamic_cast<core::topology::BaseTopologyObject *>(obj) )
            flags |= 1 << TOPOLOGY;
        if (dynamic_cast<core::loader::BaseLoader*>(obj))
            flags |= 1 << LOADER;
        if (dynamic_cast<core::objectmodel::ConfigurationSetting*>(obj))
            flags |= 1 << CONFIGURATIONSETTING;
        if (dynamic_cast<core::visual::VisualModel*>(obj) && !flags)
            flags |= 1 << VMODEL;
        if (!flags)
            flags |= 1 << OBJECT;
    }
    else return NULL;

    static std::map<unsigned int, QPixmap*> pixmaps;
    if (!pixmaps.count(flags))
    {
        int nc = 0;
        for (int i=0; i<ALLCOLORS; i++)
            if (flags & (1<<i))
                ++nc;
        int nx = 2+iconWidth*nc+iconMargin;
        QImage * img = new QImage(nx,iconHeight,32);
        img->setAlphaBuffer(true);
        img->fill(qRgba(0,0,0,0));
        // Workaround for qt 3.x where fill() does not set the alpha channel
        for (int y=0 ; y < iconHeight ; y++)
            for (int x=0 ; x < nx ; x++)
                img->setPixel(x,y,qRgba(0,0,0,0));

        for (int y=0 ; y < iconHeight ; y++)
            img->setPixel(0,y,qRgba(0,0,0,255));
        nc = 0;
        for (int i=0; i<ALLCOLORS; i++)
            if (flags & (1<<i))
            {
                int x0 = 1+iconWidth*nc;
                int x1 = x0+iconWidth-1;
                //QColor c(COLOR[i]);
                const char* color = COLOR[i];
                //c.setAlpha(255);
                int r = (hexval(color[1])*16+hexval(color[2]));
                int g = (hexval(color[3])*16+hexval(color[4]));
                int b = (hexval(color[5])*16+hexval(color[6]));
                int a = 255;
                for (int x=x0; x <=x1 ; x++)
                {
                    img->setPixel(x,0,qRgba(0,0,0,255));
                    img->setPixel(x,iconHeight-1,qRgba(0,0,0,255));
                    for (int y=1 ; y < iconHeight-1 ; y++)
                        //img->setPixel(x,y,c.value());
                        img->setPixel(x,y,qRgba(r,g,b,a));
                }
                //bitBlt(img,nimg*(iconWidth+2),0,classIcons,iconMargin,iconPos[i],iconWidth,iconHeight);
                ++nc;
            }
        for (int y=0 ; y < iconHeight ; y++)
            img->setPixel(2+iconWidth*nc-1,y,qRgba(0,0,0,255));
#ifdef SOFA_QT4
        pixmaps[flags] = new QPixmap(QPixmap::fromImage(*img));
#else
        pixmaps[flags] = new QPixmap(*img);
#endif
        delete img;
    }
    return pixmaps[flags];
}



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
void GraphListenerQListView::addChild(Node* parent, Node* child)
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
        else
        {
            //Node with multiple parent
            Q3ListViewItem *nodeItem=items[child];
            if (parent &&
                parent != findObject(nodeItem->parent()) &&
                !nodeWithMultipleParents.count(nodeItem))
            {
                Q3ListViewItem* item= createItem(items[parent]);
                item->setDropEnabled(true);
                QString name=QString("MultiNode ") + QString(child->getName().c_str());
                item->setText(0, name);
                nodeWithMultipleParents.insert(std::make_pair(items[child], item));
                static QPixmap pixMultiNode((const char**)iconmultinode_xpm);
                item->setPixmap(0, pixMultiNode);
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

        //	    if (std::string(child->getName(),0,7) != "default")
        item->setDropEnabled(true);
        item->setText(0, child->getName().c_str());
        if (child->getWarnings().empty())
        {
            QPixmap* pix = getPixmap(child);
            if (pix)
                item->setPixmap(0, *pix);
        }
        else
        {
            static QPixmap pixWarning((const char**)iconwarning_xpm);
            item->setPixmap(0,pixWarning);
        }

        item->setOpen(true);
        items[child] = item;
    }
    // Add all objects and grand-children
    MutationListener::addChild(parent, child);
}

/*****************************************************************************************************************/
void GraphListenerQListView::removeChild(Node* parent, Node* child)
{
    MutationListener::removeChild(parent, child);
    if (items.count(child))
    {
        delete items[child];
        items.erase(child);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::moveChild(Node* previous, Node* parent, Node* child)
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
void GraphListenerQListView::addObject(Node* parent, core::objectmodel::BaseObject* object)
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
        if (!dynamic_cast<core::objectmodel::ConfigurationSetting*>(object))
        {
            name += "  ";
            name += object->getName();
        }
        item->setText(0, name.c_str());

        if (object->getWarnings().empty())
        {
            QPixmap* pix = getPixmap(object);
            if (pix)
                item->setPixmap(0, *pix);

        }
        else
        {
            static QPixmap pixWarning((const char**)iconwarning_xpm);
            item->setPixmap(0,pixWarning);
        }


        items[object] = item;
    }
    // Add all slaves
    MutationListener::addObject(parent, object);
}


/*****************************************************************************************************************/
void GraphListenerQListView::removeObject(Node* parent, core::objectmodel::BaseObject* object)
{
    // Remove all slaves
    MutationListener::removeObject(parent, object);
    if (items.count(object))
    {
        delete items[object];
        items.erase(object);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::moveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object)
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
void GraphListenerQListView::addSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    if (frozen) return;
    if (items.count(slave))
    {
        Q3ListViewItem* item = items[slave];
        if (item->listView() == NULL)
        {
            if (items.count(master))
                items[master]->insertItem(item);
            else
            {
                std::cerr << "Graph -> QT ERROR: Unknown master node "<<master->getName()<<std::endl;
                return;
            }
        }
    }
    else
    {
        Q3ListViewItem* item;
        if (items.count(master))
            item = createItem(items[master]);
        else
        {
            std::cerr << "Graph -> QT ERROR: Unknown master node "<<master->getName()<<std::endl;
            return;
        }
        std::string name = sofa::helper::gettypename(typeid(*slave));
        std::string::size_type pos = name.find('<');
        if (pos != std::string::npos)
            name.erase(pos);
        if (!dynamic_cast<core::objectmodel::ConfigurationSetting*>(slave))
        {
            name += "  ";
            name += slave->getName();
        }
        item->setText(0, name.c_str());

        if (slave->getWarnings().empty())
        {
            QPixmap* pix = getPixmap(slave);
            if (pix)
                item->setPixmap(0, *pix);

        }
        else
        {
            static QPixmap pixWarning((const char**)iconwarning_xpm);
            item->setPixmap(0,pixWarning);
        }


        items[slave] = item;
    }
    // Add all slaves
    MutationListener::addSlave(master, slave);
}


/*****************************************************************************************************************/
void GraphListenerQListView::removeSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    // Remove all slaves
    MutationListener::removeSlave(master, slave);
    if (items.count(slave))
    {
        delete items[slave];
        items.erase(slave);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::moveSlave(core::objectmodel::BaseObject* previous, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    if (frozen && items.count(slave))
    {
        Q3ListViewItem* itemSlave = items[slave];
        Q3ListViewItem* itemPrevious = items[previous];
        itemPrevious->takeItem(itemSlave);
        return;
    }
    if (!items.count(slave) || !items.count(previous))
    {
        addSlave(master, slave);
    }
    else if (!items.count(master))
    {
        removeSlave(previous, slave);
    }
    else
    {
        Q3ListViewItem* itemSlave = items[slave];
        Q3ListViewItem* itemPrevious = items[previous];
        Q3ListViewItem* itemMaster = items[master];
        itemPrevious->takeItem(itemSlave);
        itemMaster->insertItem(itemSlave);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::sleepChanged(Node* node)
{
	if (items.count(node))
    {
		Q3ListViewItem* item = items[node];
		QPixmap* pix = getPixmap(node);
		if (pix)
			item->setPixmap(0, *pix);
	}
}

/*****************************************************************************************************************/
void GraphListenerQListView::freeze(Node* groot)
{
    if (!items.count(groot)) return;
    frozen = true;
}


/*****************************************************************************************************************/
void GraphListenerQListView::unfreeze(Node* groot)
{
    if (!items.count(groot)) return;
    frozen = false;
    addChild(NULL, groot);
}

/*****************************************************************************************************************/
core::objectmodel::Base* GraphListenerQListView::findObject(const Q3ListViewItem* item)
{
    core::objectmodel::Base* base = NULL;

    if(item)
    {
        for ( std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it = items.begin() ; it != items.end() ; ++ it )
        {
            if ( ( *it ).second == item )
            {
                base = (*it).first;
                return base;
            }
        }
    }
    if (!base) //Can be a multi node
    {
        std::multimap<Q3ListViewItem *, Q3ListViewItem*>::iterator it;
        for (it=nodeWithMultipleParents.begin(); it!=nodeWithMultipleParents.end(); ++it)
        {
            if (it->second == item) return findObject(it->first);
        }
    }
    return base;
}

/*****************************************************************************************************************/
core::objectmodel::BaseData* GraphListenerQListView::findData(const Q3ListViewItem* item)
// returns NULL if nothing is found.
{
    BaseData* data = NULL;
    if(item)
    {
        std::map<BaseData*,Q3ListViewItem*>::const_iterator it;
        for( it = datas.begin(); it != datas.end(); ++it)
        {
            if((*it).second == item)
            {
                data = (*it).first;
            }
        }
    }
    return data;
}
/*****************************************************************************************************************/
void GraphListenerQListView::removeDatas(core::objectmodel::BaseObject* parent)
{

    BaseData* data = NULL;
    std::string name;
    if (frozen) return;

    if( items.count(parent) )
    {
        const sofa::core::objectmodel::Base::VecData& fields = parent->getDataFields();
        for( sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin();
                it != fields.end();
                ++it)
        {
            data = (*it);
            if(datas.count(data))
            {
                delete datas[data];
                datas.erase(data);
            }
        }
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::addDatas(sofa::core::objectmodel::BaseObject *parent)
{
    if (frozen) return;
    Q3ListViewItem* new_item;
    std::string name;
    BaseData* data = NULL;
    if(items.count(parent))
    {
        const sofa::core::objectmodel::Base::VecData& fields = parent->getDataFields();
        for( sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin();
                it!=fields.end();
                ++it)
        {
            data = (*it);
            if(!datas.count(data))
            {
                static QPixmap pixData((const char**)icondata_xpm);
                new_item = createItem(items[parent]);
                name += "  ";
                name += data->getName();
                datas.insert(std::pair<BaseData*,Q3ListViewItem*>(data,new_item));
                new_item->setText(0, name.c_str());
                new_item->setPixmap(0,pixData);
                widget->ensureItemVisible(new_item);
                name.clear();
            }
        }
    }
}




} //qt
} //gui
} //sofa
