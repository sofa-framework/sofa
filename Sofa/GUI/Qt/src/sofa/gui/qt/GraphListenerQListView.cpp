/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <QApplication>
#include <sofa/simulation/Colors.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/component/sceneutility/InfoComponent.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/Mapping.h>
#include <sofa/simulation/Node.h>
#include <bitset>

using sofa::component::sceneutility::InfoComponent ;
using namespace sofa::simulation;

#include "resources/icons/iconmultinode.xpm"
#include "resources/icons/iconnode.xpm"
#include "resources/icons/iconinfo.xpm"
#include "resources/icons/iconwarning.xpm"
#include "resources/icons/iconerror.xpm"
#include "resources/icons/icondata.xpm"
#include "resources/icons/iconsleep.xpm"


#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::Message ;

using namespace sofa::core::objectmodel;

namespace sofa::gui::qt
{

//***********************************************************************************************************
static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

void addOverlayToImage(QImage* image, const QImage* overlay, int width, int height)
{
    for (int x=0;x<width;x++)
    {
        for(int y=0;y<height;y++)
        {
            if( qAlpha(overlay->pixel(x,y)) == 255 )
                image->setPixel(x, y,  overlay->pixel(x,y) );
        }
    }
}

/// get the appropriate pixmap for a given state. If the pixmap does not exists, then
/// the appropriate one is generated and stored in a cache for further re-use.
QPixmap* getPixmapForFlags(unsigned int flags, std::map<unsigned int, QPixmap*>& pixmaps)
{
    // is there is not pixmap for such state, then generates one
    if (pixmaps.find(flags) == pixmaps.end())
    {
        if(flags&(1<<Colors::NODE))
        {
            const char** icon = nullptr;
            icon = reinterpret_cast<const char**>(iconnode_xpm);

            // Create a new image from pixmap
            const QImage timg(icon);
            QImage image {timg.convertToFormat(QImage::Format_ARGB32)};
            pixmaps[flags] = new QPixmap(QPixmap::fromImage(image));
        }else if(flags & (1 << Colors::OBJECT))
        {
            const int iconWidth=16;
            const int iconHeight=16;
            const int iconMargin=5;

            QImage img { iconWidth,iconHeight,QImage::Format_ARGB32 };
            img.fill(qRgba(0,0,0,0));

            // draw vertical black lines
            for (int y=iconMargin ; y < iconHeight-1 ; y++)
            {
                img.setPixel(0,y,qRgba(0,0,0,255));
                img.setPixel(iconWidth-iconMargin,y,qRgba(0,0,0,255));
            }

            // draw vertical black horizontal lines
            for (int x=0 ; x < iconWidth-iconMargin ; x++)
            {
                img.setPixel(x,iconMargin,qRgba(0,0,0,255));
                img.setPixel(x,iconHeight-1,qRgba(0,0,0,255));
            }

            // select the largest "type" for color.
            int nc = 0;
            for (int i=0; i<Colors::ALLCOLORS; i++)
            {
                if (flags & (1<<i))
                {
                    nc=i;
                }
            }

            const char* color = Colors::COLOR[nc];
            const int r = (hexval(color[1])*16+hexval(color[2]));
            const int g = (hexval(color[3])*16+hexval(color[4]));
            const int b = (hexval(color[5])*16+hexval(color[6]));
            const int a = 255;

            // draw colored square
            for (int x=1 ; x < iconWidth-iconMargin ; x++)
            {
                for(int y=iconMargin+1 ; y < iconHeight-1; y++)
                {
                    img.setPixel(x,y, qRgba(r,g,b,a));
                }
            }
            pixmaps[flags] = new QPixmap(QPixmap::fromImage(img));
        }else
        {
            throw std::runtime_error("Missing type in flags " + std::to_string(flags));
        }
    }
    return pixmaps[flags];
}

QPixmap* getPixmapForFlagsAndState(unsigned int typeFlags,
                                   bool isSleeping, bool haveInfo, bool haveWarning, bool haveErrors)
{
    static QPixmap pixSleeping(reinterpret_cast<const char**>(iconsleep_xpm));
    static QImage imgSleeping = pixSleeping.toImage();

    static QPixmap pixInfo(reinterpret_cast<const char**>(iconinfo_xpm));
    static QImage imgInfo8 = pixInfo.scaledToWidth(16).toImage();

    static QPixmap pixError(reinterpret_cast<const char**>(iconerror_xpm));
    static QImage imgError8 = pixError.scaledToWidth(16).toImage();

    static QPixmap pixWarning(reinterpret_cast<const char**>(iconwarning_xpm));
    static QImage imgWarning8 = pixWarning.scaledToWidth(16).toImage();

    static std::map<unsigned int, QPixmap*> pixmaps;

    unsigned int flags=typeFlags;
    if(isSleeping)
        flags |= 1 << (Colors::ALLCOLORS+1);

    if(haveInfo)
        flags |= 1 << (Colors::ALLCOLORS+2);

    if(haveWarning)
        flags |= 1 << (Colors::ALLCOLORS+3);

    if(haveErrors)
        flags |= 1 << (Colors::ALLCOLORS+4);

    // is there is a pixmap for such type+state, if so then returns it
    if (pixmaps.find(flags) != pixmaps.end())
        return pixmaps[flags];

    // if there is no such pixmap creates a new one by combining the object type with its status as overlay.

    // get the background pixmap
    auto timg = getPixmapForFlags(typeFlags, pixmaps)->toImage() ;
    QImage image {timg.convertToFormat(QImage::Format_ARGB32)};

    // generates one our of the base flags with the state overlay
    if(isSleeping)
        addOverlayToImage(&image, &imgSleeping, imgSleeping.width(), imgSleeping.height());

    const QImage* overlay=nullptr;
    if( haveInfo )
        overlay = &imgInfo8;
    if( haveWarning )
        overlay = &imgWarning8;
    if( haveErrors )
        overlay = &imgError8;

    if(overlay)
        addOverlayToImage(&image, overlay, 16, 16);

    pixmaps[flags] = new QPixmap(QPixmap::fromImage(image));
    return pixmaps[flags];
}


QPixmap* getBaseNodePixmap(core::objectmodel::BaseNode* node, bool haveInfo, bool haveWarning, bool haveErrors)
{
    unsigned int flags = 1 << Colors::NODE;
    return getPixmapForFlagsAndState(flags,
                                     node->getContext()->isSleeping(),
                                     haveInfo, haveWarning, haveErrors);
}

QPixmap* getBaseObjectPixmap(core::objectmodel::BaseObject* obj, bool haveInfo, bool haveWarning, bool haveErrors)
{
    unsigned int flags = 0;

    flags |= 1 << Colors::OBJECT;
    if (obj->toContextObject())
        flags |= 1 << Colors::CONTEXT;
    if (obj->toBehaviorModel())
        flags |= 1 << Colors::BMODEL;
    if (obj->toCollisionModel())
        flags |= 1 << Colors::CMODEL;
    if (obj->toBaseMechanicalState())
        flags |= 1 << Colors::MMODEL;
    if (obj->toBaseProjectiveConstraintSet())
        flags |= 1 << Colors::PROJECTIVECONSTRAINTSET;
    if (obj->toBaseConstraintSet())
        flags |= 1 << Colors::CONSTRAINTSET;
    if (obj->toBaseInteractionForceField() &&
        obj->toBaseInteractionForceField()->getMechModel1()!=obj->toBaseInteractionForceField()->getMechModel2())
        flags |= 1 << Colors::IFFIELD;
    else if (obj->toBaseForceField())
        flags |= 1 << Colors::FFIELD;
    if (obj->toBaseAnimationLoop()
        || obj->toOdeSolver())
        flags |= 1 << Colors::SOLVER;
    if (obj->toPipeline()
        || obj->toIntersection()
        || obj->toDetection()
        || obj->toContactManager()
        || obj->toCollisionGroupManager())
        flags |= 1 << Colors::COLLISION;
    if (obj->toBaseMapping() && obj->toBaseMapping()->isMechanical())
        flags |= (1 << Colors::MMAPPING);
    if (obj->toBaseMapping())
        flags |= (1 << Colors::MAPPING);
    if (obj->toBaseMass())
        flags |= 1 << Colors::MASS;
    if (obj->toTopology ()
        || obj->toBaseTopologyObject() )
        flags |= 1 << Colors::TOPOLOGY;
    if (obj->toBaseLoader())
        flags |= 1 << Colors::LOADER;
    if (obj->toConfigurationSetting())
        flags |= 1 << Colors::CONFIGURATIONSETTING;
    if (obj->toVisualModel())
        flags |= 1 << Colors::VMODEL;

    return getPixmapForFlagsAndState(flags, false, haveInfo, haveWarning, haveErrors);
}


QPixmap* getPixmap(core::objectmodel::Base* obj, bool haveInfo, bool haveWarning, bool haveErrors)
{
    if (obj->toBaseNode())
        return getBaseNodePixmap(obj->toBaseNode(), haveInfo,haveWarning, haveErrors);

    if (obj->toBaseObject())
        return getBaseObjectPixmap(obj->toBaseObject(), haveInfo,haveWarning, haveErrors);

    return nullptr;
}

void setMessageIconFrom(QTreeWidgetItem* item, Base* object)
{
    const bool haveInfos = object->countLoggedMessages({Message::Info, Message::Deprecated, Message::Advice})!=0;
    const bool haveWarnings = object->countLoggedMessages({Message::Warning})!=0;
    const bool haveErrors = object->countLoggedMessages({Message::Error, Message::Fatal})!=0;

    const QPixmap* pix = getPixmap(object, haveInfos, haveWarnings, haveErrors);
    if (pix)
        item->setIcon(0, QIcon(*pix));
}

ObjectStateListener::ObjectStateListener(
        QTreeWidgetItem* item_,
        sofa::core::objectmodel::Base* object_) : item(item_), object(object_)
{
    // We want the view to react to a change in the message log
    object->d_messageLogCount.addOutput(this);

    // We want the view to react to a change in the name
    object->name.addOutput(this);
}

ObjectStateListener::~ObjectStateListener()
{
    object->d_messageLogCount.delOutput(this);
    object->name.delOutput(this);
}

void ObjectStateListener::update() {}
void ObjectStateListener::notifyEndEdit()
{
    setMessageIconFrom(item, object.get());

    const QString oldName = item->text(0);
    const QString newName = QString::fromStdString(object->getName());
    if(newName != oldName)
        item->setText(0, newName);
}

GraphListenerQListView::~GraphListenerQListView()
{
    for(auto [key, listener] : listeners)
    {
        delete listener;
    }
    listeners.clear();
}

/*****************************************************************************************************************/
QTreeWidgetItem* GraphListenerQListView::createItem(QTreeWidgetItem* parent)
{
    if(parent->childCount() == 0)
        return new QTreeWidgetItem(parent);
    return new QTreeWidgetItem(parent, parent->child(parent->childCount()-1));
}

/*****************************************************************************************************************/
void GraphListenerQListView::onBeginAddChild(Node* parent, Node* child)
{
    if (widget->isLocked())
    {
        widget->setViewToDirty();
        return;
    }
    if (items.count(child))
    {
        QTreeWidgetItem* item = items[child];
        if (item->treeWidget() == nullptr)
        {
            if (parent == nullptr)
            {
                dmsg_info("GraphListenerQListView") << "CREATING TOP LEVEL NODE '"<<child->getName()<<"'";
                widget->insertTopLevelItem(0, item);
            }
            else if (items.count(parent))
            {
                items[parent]->insertChild(0, item);
            }
            else
            {
                dmsg_error("GraphListenerQListView") << "Unknown parent node '"<<parent->getName()<<"'";
                return;
            }
        }
        else
        {
            static QPixmap pixMultiNode(reinterpret_cast<const char**>(iconmultinode_xpm));

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
                    }
                }
            }
        }
    }
    else
    {
        QTreeWidgetItem* item;
        if (parent == nullptr)
            item = new QTreeWidgetItem(widget);
        else if (items.count(parent))
        {
            item = createItem(items[parent]);
        }
        else
        {
            dmsg_error("GraphListenerQListView") << "Unknown parent node '"<<parent->getName()<<"'";
            return;
        }

        item->setText(0, child->getName().c_str());
        item->setText(1, child->getClassName().c_str());
        item->setForeground(1, nameColor);
        QFont font = QApplication::font();
        font.setBold(true);
        item->setFont(0, font);
        setMessageIconFrom(item, child);

        item->setExpanded(true);
        items[child] = item;

        // Add a listener to connect changes on the component state with its graphical view.
        listeners[child] = new ObjectStateListener(item, child);
    }

    for (BaseObject::SPtr obj : child->object)
        onBeginAddObject(child, obj.get());
    for (Node::SPtr node : child->child)
        onBeginAddChild(child, node.get());
}

/*****************************************************************************************************************/
void GraphListenerQListView::onBeginRemoveChild(Node* parent, Node* child)
{
    SOFA_UNUSED(parent);
    for (Node::ObjectIterator it = child->object.begin(); it != child->object.end(); ++it)
        onBeginRemoveObject(child, it->get());
    for (Node::ChildIterator it = child->child.begin(); it != child->child.end(); ++it)
        onBeginRemoveChild(child, it->get());

    if (items.count(child))
    {
        delete items[child];
        delete listeners[child];
        items.erase(child);
        listeners.erase(child);
    }
}


/*****************************************************************************************************************/
void GraphListenerQListView::onBeginAddObject(Node* parent, core::objectmodel::BaseObject* object)
{
    if(widget->isLocked())
    {
        widget->setViewToDirty();
        return;
    }
    if (items.count(object))
    {
        QTreeWidgetItem* item = items[object];
        if (item->treeWidget() == nullptr)
        {
            if (items.count(parent))
                //                items[parent]->insertItem(item);
                items[parent]->addChild(item);
            else
            {
                dmsg_error("GraphListenerQListView") << "Unknown parent node " << parent->getName()<< "'";
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
            dmsg_error("GraphListenerQListView") << "Unknown parent node " << parent->getName()<< "'";
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
            name = object->getName() ;
            item->setText(1, object->getClassName().c_str());
            item->setForeground(1, nameColor);
            const QString tooltip( ("Name: " + name + "\nClass Name: " + object->getClassName()).c_str());
            item->setToolTip(0, tooltip);
            item->setToolTip(1, tooltip);
        }

        item->setText(0, name.c_str());

        setMessageIconFrom(item, object);

        items[object] = item;
        listeners[object] = new ObjectStateListener(item, object);
    }
    for (BaseObject::SPtr slave : object->getSlaves())
        onBeginAddSlave(object, slave.get());
}


/*****************************************************************************************************************/
void GraphListenerQListView::onBeginRemoveObject(Node* parent, core::objectmodel::BaseObject* object)
{
    SOFA_UNUSED(parent);
    for (BaseObject::SPtr slave : object->getSlaves())
        onBeginRemoveSlave(object, slave.get());

    if (items.count(object))
    {
        delete items[object];
        items.erase(object);

        delete listeners[object];
        listeners.erase(object);
    }
}


/*****************************************************************************************************************/
void GraphListenerQListView::onBeginAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    if(widget->isLocked())
    {
        widget->setViewToDirty();
        return;
    }
    if (items.count(slave))
    {
        QTreeWidgetItem* item = items[slave];
        if (item->treeWidget() == nullptr)
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
        std::string className = sofa::helper::gettypename(typeid(*slave));
        if (const std::string::size_type pos = className.find('<'); pos != std::string::npos)
            className.erase(pos);
        if (!slave->toConfigurationSetting())
        {
            const auto& name = slave->getName();
            item->setText(0, name.c_str());
            item->setForeground(1, nameColor);

            const QString tooltip( ("Name: " + name + "\nClass Name: " + className).c_str());
            item->setToolTip(0, tooltip);
            item->setToolTip(1, tooltip);
        }
        item->setText(1, className.c_str());

        setMessageIconFrom(item, slave);

        items[slave] = item;
    }

    const core::objectmodel::BaseObject::VecSlaves& slaves = slave->getSlaves();
    for (unsigned int i=0; i<slaves.size(); ++i)
        onBeginAddSlave(slave, slaves[i].get());
}


/*****************************************************************************************************************/
void GraphListenerQListView::onBeginRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    SOFA_UNUSED(master);
    const core::objectmodel::BaseObject::VecSlaves& slaves = slave->getSlaves();
    for (unsigned int i=0; i<slaves.size(); ++i)
        onBeginRemoveSlave(slave, slaves[i].get());

    if (items.count(slave))
    {
        delete items[slave];
        items.erase(slave);

        delete listeners[slave];
        listeners.erase(slave);
    }
}

/*****************************************************************************************************************/
void GraphListenerQListView::sleepChanged(Node* node)
{
    if (items.count(node))
    {
        QTreeWidgetItem* item = items[node];
        const QPixmap* pix = getPixmap(node,false,false,false);
        if (pix)
            item->setIcon(0, QIcon(*pix));
    }
}

/*****************************************************************************************************************/
core::objectmodel::Base* GraphListenerQListView::findObject(const QTreeWidgetItem* item)
{
    core::objectmodel::Base* base = nullptr;

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
// returns nullptr if nothing is found.
{
    BaseData* data = nullptr;
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
    if (widget->isLocked())
    {
        widget->setViewToDirty();
        return;
    }

    BaseData* data = nullptr;
    std::string name;

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
    if (widget->isLocked())
    {
        widget->setViewToDirty();
        return;
    }

    QTreeWidgetItem* new_item;
    std::string name;
    BaseData* data = nullptr;
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
                static QPixmap pixData(reinterpret_cast<const char**>(icondata_xpm));
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

} //namespace sofa::gui::qt
