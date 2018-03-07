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

#include "GraphListenerQListView.h"
#include <sofa/simulation/Colors.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <SofaComponentBase/InfoComponent.h>
using sofa::component::InfoComponent ;

#include "resources/icons/iconmultinode.xpm"
#include "resources/icons/iconnode.xpm"
#include "resources/icons/iconinfo.xpm"
#include "resources/icons/iconwarning.xpm"
#include "resources/icons/iconerror.xpm"
#include "resources/icons/icondata.xpm"
#include "resources/icons/iconsleep.xpm"


#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::Message ;

namespace sofa
{
using namespace core::objectmodel;
namespace gui
{

namespace qt
{
//***********************************************************************************************************

static const int iconWidth=8;
static const int iconHeight=16;
static const int iconMargin=6;

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

const std::string getClass(core::objectmodel::Base* obj){
    if (obj->toBaseNode())
    {
        return "Node";
    }
    if (obj->toBaseObject())
    {
        if (obj->toContextObject())
            return "Context";
        if (obj->toBehaviorModel())
            return "BehaviorModel";
        if (obj->toCollisionModel())
            return "CollisionModel";
        if (obj->toBaseMechanicalState())
            return "MechanicalModel";
        if (obj->toBaseProjectiveConstraintSet())
            return "ProjectiveConstraintSet";
        if (obj->toBaseConstraintSet())
            return "BaseConstraintSet";
        if (obj->toBaseInteractionForceField() &&
                obj->toBaseInteractionForceField()->getMechModel1()!=obj->toBaseInteractionForceField()->getMechModel2())
            return "InteractionForceField";
        if (obj->toBaseForceField())
            return "ForceField";
        if (obj->toBaseAnimationLoop()
                || obj->toOdeSolver())
            return "Solver";
        if (obj->toPipeline()
                || obj->toIntersection()
                || obj->toDetection()
                || obj->toContactManager()
                || obj->toCollisionGroupManager())
            return "Collision";
        if (obj->toBaseMapping())
            return "Mapping";
        if (obj->toBaseMass())
            return "Mass";
        if (obj->toTopology ()
                || obj->toBaseTopologyObject() )
            return "Topology";
        if (obj->toBaseLoader())
            return "Loader";
        if (obj->toConfigurationSetting())
            return "Configuration";
        if (obj->toVisualModel())
            return "Visual";
    }
    return "Other";
}



QPixmap* getPixmap(core::objectmodel::Base* obj, bool haveInfo, bool haveWarning, bool haveErrors)
{
    static QPixmap pixInfo((const char**)iconinfo_xpm);
    static QImage imgInfo8 = pixInfo.scaledToWidth(16).toImage();

    static QPixmap pixError((const char**)iconerror_xpm);
    static QImage imgError8 = pixError.scaledToWidth(16).toImage();

    static QPixmap pixWarning((const char**)iconwarning_xpm);
    static QImage imgWarning8 = pixWarning.scaledToWidth(16).toImage();


    using namespace sofa::simulation::Colors;
    unsigned int flags=0;

    if (obj->toBaseNode())
    {
        int flags = 0 ;
        const char** icon = (const char**)iconsleep_xpm ;
        if( !obj->toBaseNode()->getContext()->isSleeping() ){
            icon = (const char**)iconnode_xpm ;
            flags = 1 ;
        }

        if(haveInfo)
            flags |= 1 << (2) ;

        if(haveWarning)
            flags |= 1 << (3) ;

        if(haveErrors)
            flags |= 1 << (4) ;


        static std::map<unsigned int, QPixmap*> pixmaps;
        if (!pixmaps.count(flags))
        {
            /// Create a new image from pixmap
            QImage timg(icon) ;
            QImage* img = new QImage(timg.convertToFormat(QImage::Format_ARGB32)) ;

            QImage* overlaysymbol=nullptr;
            if( haveInfo )
                overlaysymbol = &imgInfo8 ;
            if( haveWarning )
                overlaysymbol = &imgWarning8 ;
            if( haveErrors )
                overlaysymbol = &imgError8 ;

            if(overlaysymbol){
                for (int x=0;x<16;x++)
                {
                    for(int y=0;y<16;y++)
                    {
                        if( qAlpha(overlaysymbol->pixel(x,y)) == 255 ){
                            img->setPixel(x, y,  overlaysymbol->pixel(x,y) );
                        }
                    }
                }
            }
            pixmaps[flags] = new QPixmap(QPixmap::fromImage(*img));
        }

        return pixmaps[flags] ;
    }
    else if (obj->toBaseObject())
    {
        if (obj->toContextObject())
            flags |= 1 << CONTEXT;
        if (obj->toBehaviorModel())
            flags |= 1 << BMODEL;
        if (obj->toCollisionModel())
            flags |= 1 << CMODEL;
        if (obj->toBaseMechanicalState())
            flags |= 1 << MMODEL;
        if (obj->toBaseProjectiveConstraintSet())
            flags |= 1 << PROJECTIVECONSTRAINTSET;
        if (obj->toBaseConstraintSet())
            flags |= 1 << CONSTRAINTSET;
        if (obj->toBaseInteractionForceField() &&
                obj->toBaseInteractionForceField()->getMechModel1()!=obj->toBaseInteractionForceField()->getMechModel2())
            flags |= 1 << IFFIELD;
        else if (obj->toBaseForceField())
            flags |= 1 << FFIELD;
        if (obj->toBaseAnimationLoop()
                || obj->toOdeSolver())
            flags |= 1 << SOLVER;
        if (obj->toPipeline()
                || obj->toIntersection()
                || obj->toDetection()
                || obj->toContactManager()
                || obj->toCollisionGroupManager())
            flags |= 1 << COLLISION;
        if (obj->toBaseMapping())
            flags |= 1 << ((obj->toBaseMapping())->isMechanical()?MMAPPING:MAPPING);
        if (obj->toBaseMass())
            flags |= 1 << MASS;
        if (obj->toTopology ()
                || obj->toBaseTopologyObject() )
            flags |= 1 << TOPOLOGY;
        if (obj->toBaseLoader())
            flags |= 1 << LOADER;
        if (obj->toConfigurationSetting())
            flags |= 1 << CONFIGURATIONSETTING;
        if (obj->toVisualModel() && !flags)
            flags |= 1 << VMODEL;
        if (!flags)
            flags |= 1 << OBJECT;
    }
    else return NULL;

    if(haveInfo)
        flags |= 1 << (ALLCOLORS+1) ;

    if(haveWarning)
        flags |= 1 << (ALLCOLORS+1) ;

    if(haveErrors)
        flags |= 1 << (ALLCOLORS+1) ;

    static std::map<unsigned int, QPixmap*> pixmaps;
    if (!pixmaps.count(flags))
    {
        int nc = 0;
        for (int i=0; i<ALLCOLORS; i++)
            if (flags & (1<<i))
                ++nc;
        int nx = 2+iconWidth*nc+iconMargin;
        //QImage * img = new QImage(nx,iconHeight,32);
        QImage * img = new QImage(nx,iconHeight,QImage::Format_ARGB32);

        //img->setAlphaBuffer(true);
        img->fill(qRgba(0,0,0,0));
        // Workaround for qt 3.x where fill() does not set the alpha channel
        for (int y=0 ; y < iconHeight ; y++)
            for (int x=0 ; x < nx ; x++)
                img->setPixel(x,y,qRgba(0,0,0,0));

        // left Line
        for (int y=iconMargin ; y < iconHeight ; y++)
            img->setPixel(0,y,qRgba(0,0,0,255));

        nc = 0;
        for (int i=0; i<ALLCOLORS; i++)
            if (flags & (1<<i))
            {
                int x0 = 1+iconWidth*nc;
                int x1 = x0+iconWidth-1;
                const char* color = COLOR[i];
                int r = (hexval(color[1])*16+hexval(color[2]));
                int g = (hexval(color[3])*16+hexval(color[4]));
                int b = (hexval(color[5])*16+hexval(color[6]));
                int a = 255;
                for (int x=x0; x <=x1 ; x++)
                {
                    img->setPixel(x,iconMargin-1,qRgba(0,0,0,255));
                    img->setPixel(x,iconHeight-1,qRgba(0,0,0,255));
                    for (int y=iconMargin ; y < iconHeight-1 ; y++)
                        img->setPixel(x,y,qRgba(r,g,b,a));
                }
                ++nc;
            }

        // right line Line
        for (int y=iconMargin ; y < iconHeight ; y++)
            img->setPixel(2+iconWidth*nc-1,y,qRgba(0,0,0,255));

        QImage* overlaysymbol=nullptr;
        if( haveInfo )
            overlaysymbol = &imgInfo8 ;
        if( haveWarning )
            overlaysymbol = &imgWarning8 ;
        if( haveErrors )
            overlaysymbol = &imgError8 ;

        if(overlaysymbol){
            for (int x=0;x<16;x++)
            {
                for(int y=0;y<16;y++)
                {
                    if( qAlpha(overlaysymbol->pixel(x,y)) == 255 )
                        img->setPixel(x, y,  overlaysymbol->pixel(x,y) );
                }
            }
        }

        pixmaps[flags] = new QPixmap(QPixmap::fromImage(*img));

        delete img;
    }
    return pixmaps[flags];
}

void setMessageIconFrom(QTreeWidgetItem* item, Base* object)
{
    bool haveInfos = object->countLoggedMessages({Message::Info, Message::Deprecated, Message::Advice})!=0;
    bool haveWarnings = object->countLoggedMessages({Message::Warning})!=0;
    bool haveErrors = object->countLoggedMessages({Message::Error, Message::Fatal})!=0;

    QPixmap* pix = getPixmap(object, haveInfos, haveWarnings, haveErrors);
    if (pix)
        item->setIcon(0, QIcon(*pix));
}

/*****************************************************************************************************************/
QTreeWidgetItem* GraphListenerQListView::createItem(QTreeWidgetItem* parent)
{
    //    QTreeWidgetItem* last = parent->firstChild();
    //    if (last == NULL)
    //        return new QTreeWidgetItem(parent);
    //    while (last->nextSibling()!=NULL)
    //        last = last->nextSibling();
    //    return new QTreeWidgetItem(parent, last);
    if(parent->childCount() == 0)
        return new QTreeWidgetItem(parent);
    return new QTreeWidgetItem(parent, parent->child(parent->childCount()-1));
}



/*****************************************************************************************************************/
void GraphListenerQListView::addChild(Node* parent, Node* child)
{

    if (frozen) return;
    if (items.count(child))
    {
        QTreeWidgetItem* item = items[child];
        if (item->treeWidget() == NULL)
        {
            if (parent == NULL)
                widget->insertTopLevelItem(0, item);
            else if (items.count(parent))
                items[parent]->insertChild(0, item);
            else
            {
                dmsg_error("GraphListenerQListView") << "Unknown parent node '"<<parent->getName()<<"'";
                return;
            }
        }
        else
        {
            static QPixmap pixMultiNode((const char**)iconmultinode_xpm);

            // Node with multiple parents
            if (parent &&
                    parent != findObject(item->parent()) )
            {
                // check that the multinode have not been added yet
                // i.e. verify that all every item equivalent to current 'item' (in nodeWithMultipleParents) do not have the same 'parent'
                std::multimap<QTreeWidgetItem *, QTreeWidgetItem*>::iterator it=nodeWithMultipleParents.lower_bound(item), itend=nodeWithMultipleParents.upper_bound(item);
                for ( ; it!=itend && it->second->parent() != items[parent] ; ++it);
                if( it==itend )
                {
                    QTreeWidgetItem* itemNew = createItem(items[parent]);
                    //itemNew->setDropEnabled(true);
                    //                QString name=QString("MultiNode ") + QString(child->getName().c_str());
                    //                itemNew->setText(0, name);
                    itemNew->setText(0, child->getName().c_str());
                    nodeWithMultipleParents.insert(std::make_pair(item, itemNew));
                    itemNew->setIcon(0, QIcon(pixMultiNode));

                    // this is one more parent, the first child item must be displayed as a multinode
                    {
                        item->setIcon(0, QIcon(pixMultiNode));
                        //                    item->setText(0, QString("MultiNode ") + item->text(0) );
                    }
                }
            }
        }
    }
    else
    {
        QTreeWidgetItem* item;
        if (parent == NULL)
            item = new QTreeWidgetItem(widget);
        else if (items.count(parent))
            item = createItem(items[parent]);
        else
        {
            dmsg_error("GraphListenerQListView") << "Unknown parent node '"<<parent->getName()<<"'";
            return;
        }

        item->setText(0, child->getName().c_str());
        setMessageIconFrom(item, child);

        item->setExpanded(true);
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
        QTreeWidgetItem* itemChild = items[child];
        if (items.count(previous)) //itemChild->listView() != NULL)
        {
            QTreeWidgetItem* itemPrevious = items[previous];
            //itemPrevious->takeItem(itemChild);
            itemPrevious->removeChild(itemChild);
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
        QTreeWidgetItem* itemChild = items[child];
        QTreeWidgetItem* itemPrevious = items[previous];
        QTreeWidgetItem* itemParent = items[parent];
        //        itemPrevious->takeItem(itemChild);
        itemPrevious->removeChild(itemChild);
        //        itemParent->insertItem(itemChild);
        itemParent->addChild(itemChild);
    }
}



/*****************************************************************************************************************/
void GraphListenerQListView::addObject(Node* parent, core::objectmodel::BaseObject* object)
{
    if (frozen) return;
    if (items.count(object))
    {
        QTreeWidgetItem* item = items[object];
        if (item->treeWidget() == NULL)
        {
            if (items.count(parent))
                //                items[parent]->insertItem(item);
                items[parent]->addChild(item);
            else
            {
                dmsg_error("GraphListenerQListView") << "Unknown parent node "<<parent->getName()<< "'";
                return;
            }
        }
    }
    else
    {
        QTreeWidgetItem* item;
        if (items.count(parent))
            item = createItem(items[parent]);
        else
        {
            dmsg_error("GraphListenerQListView") << "Unknown parent node "<<parent->getName()<< "'";
            return;
        }

        std::string name;
        if(dynamic_cast<InfoComponent*>(object))
        {
            name = object->getName() ;
        }else if(dynamic_cast<ConfigurationSetting*>(object)){
            name = object->getClassName() ;
        }else
        {
            name = object->getClassName() + " " + object->getName() ;
        }

        item->setText(0, name.c_str());

        setMessageIconFrom(item, object);

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
        QTreeWidgetItem* itemObject = items[object];
        QTreeWidgetItem* itemPrevious = items[previous];
        //        itemPrevious->takeItem(itemObject);
        itemPrevious->removeChild(itemObject);
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
        QTreeWidgetItem* itemObject = items[object];
        QTreeWidgetItem* itemPrevious = items[previous];
        QTreeWidgetItem* itemParent = items[parent];
        //        itemPrevious->takeItem(itemObject);
        itemPrevious->removeChild(itemObject);
        //        itemParent->insertItem(itemObject);
        itemParent->addChild(itemObject);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::addSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    if (frozen) return;
    if (items.count(slave))
    {
        QTreeWidgetItem* item = items[slave];
        if (item->treeWidget() == NULL)
        {
            if (items.count(master))
                //                items[master]->insertItem(item);
                items[master]->addChild(item);
            else
            {
                dmsg_error("GraphListenerQListView") << "Unknown master node '"<<master->getName()<<"'";
                return;
            }
        }
    }
    else
    {
        QTreeWidgetItem* item;
        if (items.count(master))
            item = createItem(items[master]);
        else
        {
            dmsg_error("GraphListenerQListView") << "Unknown master node '"<<master->getName()<<"'";
            return;
        }
        std::string name = sofa::helper::gettypename(typeid(*slave));
        std::string::size_type pos = name.find('<');
        if (pos != std::string::npos)
            name.erase(pos);
        if (!slave->toConfigurationSetting())
        {
            name += "  ";
            name += slave->getName();
        }
        item->setText(0, name.c_str());

        setMessageIconFrom(item, slave);

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
        QTreeWidgetItem* itemSlave = items[slave];
        QTreeWidgetItem* itemPrevious = items[previous];
        //        itemPrevious->takeItem(itemSlave);
        itemPrevious->removeChild(itemSlave);
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
        QTreeWidgetItem* itemSlave = items[slave];
        QTreeWidgetItem* itemPrevious = items[previous];
        QTreeWidgetItem* itemMaster = items[master];
        //        itemPrevious->takeItem(itemSlave);
        itemPrevious->removeChild(itemSlave);
        //        itemMaster->insertItem(itemSlave);
        itemMaster->addChild(itemSlave);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::sleepChanged(Node* node)
{
    if (items.count(node))
    {
        QTreeWidgetItem* item = items[node];
        QPixmap* pix = getPixmap(node,false,false,false);
        if (pix)
            item->setIcon(0, QIcon(*pix));
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
core::objectmodel::Base* GraphListenerQListView::findObject(const QTreeWidgetItem* item)
{
    core::objectmodel::Base* base = NULL;

    if(item)
    {
        for ( std::map<core::objectmodel::Base*, QTreeWidgetItem* >::iterator it = items.begin() ; it != items.end() ; ++ it )
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
        std::multimap<QTreeWidgetItem *, QTreeWidgetItem*>::iterator it;
        for (it=nodeWithMultipleParents.begin(); it!=nodeWithMultipleParents.end(); ++it)
        {
            if (it->second == item) return findObject(it->first);
        }
    }
    return base;
}

/*****************************************************************************************************************/
core::objectmodel::BaseData* GraphListenerQListView::findData(const QTreeWidgetItem* item)
// returns NULL if nothing is found.
{
    BaseData* data = NULL;
    if(item)
    {
        std::map<BaseData*,QTreeWidgetItem*>::const_iterator it;
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
    QTreeWidgetItem* new_item;
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
                datas.insert(std::pair<BaseData*,QTreeWidgetItem*>(data,new_item));
                new_item->setText(0, name.c_str());
                new_item->setIcon(0, QIcon(pixData));
                //                widget->ensureItemVisible(new_item);
                widget->scrollToItem(new_item);
                name.clear();
            }
        }
    }
}




} //qt
} //gui
} //sofa
