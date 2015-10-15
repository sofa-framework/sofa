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
#include <sofa/gui/qt/DisplayFlagsDataWidget.h>
#ifdef SOFA_QT4
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLayout>
#else
#include <qlayout.h>
#endif

namespace sofa
{
namespace gui
{
namespace qt
{
using namespace sofa::core::objectmodel;
using namespace sofa::core::visual;

helper::Creator<DataWidgetFactory, DisplayFlagsDataWidget > DWClass_DisplayFlags("widget_displayFlags",true);


DisplayFlagWidget::DisplayFlagWidget(QWidget* parent, const char* name,  Qt::WFlags f ):
    Q3ListView(parent,name,f)
{
    addColumn(QString::null);
    setRootIsDecorated( TRUE );
    setTreeStepSize( 12 );
    header()->hide();
    clear();

    setMouseTracking(false);

#ifdef SOFA_QT4
    setFocusPolicy(Qt::NoFocus);
#else
    setFocusPolicy(QWidget::NoFocus);
#endif

    setFrameShadow(QFrame::Plain);
    setFrameShape(QFrame::NoFrame );

    setSortColumn(-1);
    Q3CheckListItem* itemShowAll = new Q3CheckListItem(this, "All", Q3CheckListItem::CheckBoxController);
    itemShowAll->setOpen(true);
    Q3CheckListItem* itemShowVisual    = new Q3CheckListItem(itemShowAll, "Visual", Q3CheckListItem::CheckBoxController);
    itemShowVisual->setOpen(true);
    itemShowFlag[VISUALMODELS]   = new Q3CheckListItem(itemShowVisual, "Visual Models", Q3CheckListItem::CheckBox);
    Q3CheckListItem* itemShowBehavior  = new Q3CheckListItem(itemShowAll, itemShowVisual, "Behavior", Q3CheckListItem::CheckBoxController);
    itemShowBehavior->setOpen(true);
    itemShowFlag[BEHAVIORMODELS]   = new Q3CheckListItem(itemShowBehavior,  "Behavior Models", Q3CheckListItem::CheckBox);
    itemShowFlag[FORCEFIELDS]   = new Q3CheckListItem(itemShowBehavior, itemShowFlag[BEHAVIORMODELS], "Force Fields", Q3CheckListItem::CheckBox);
    itemShowFlag[INTERACTIONFORCEFIELDS]   = new Q3CheckListItem(itemShowBehavior, itemShowFlag[FORCEFIELDS],  "Interactions", Q3CheckListItem::CheckBox);
    Q3CheckListItem* itemShowCollision = new Q3CheckListItem(itemShowAll, itemShowBehavior, "Collision", Q3CheckListItem::CheckBoxController);
    itemShowCollision->setOpen(true);
    itemShowFlag[COLLISIONMODELS]   = new Q3CheckListItem(itemShowCollision,  "Collision Models", Q3CheckListItem::CheckBox);
    itemShowFlag[BOUNDINGCOLLISIONMODELS]   = new Q3CheckListItem(itemShowCollision, itemShowFlag[COLLISIONMODELS], "Bounding Trees", Q3CheckListItem::CheckBox);
    Q3CheckListItem* itemShowMapping   = new Q3CheckListItem(itemShowAll, itemShowCollision, "Mapping", Q3CheckListItem::CheckBoxController);
    itemShowMapping->setOpen(true);
    itemShowFlag[MAPPINGS]   = new Q3CheckListItem(itemShowMapping,  "Visual Mappings", Q3CheckListItem::CheckBox);
    itemShowFlag[MECHANICALMAPPINGS]   = new Q3CheckListItem(itemShowMapping, itemShowFlag[MAPPINGS],  "Mechanical Mappings", Q3CheckListItem::CheckBox);
    Q3ListViewItem*  itemShowOptions   = new Q3ListViewItem(this, itemShowAll, "Options");
    itemShowOptions->setOpen(true);
    itemShowFlag[RENDERING]   = new Q3CheckListItem(itemShowOptions, "Advanced Rendering", Q3CheckListItem::CheckBox);
    itemShowFlag[WIREFRAME]   = new Q3CheckListItem(itemShowOptions, "Wire Frame", Q3CheckListItem::CheckBox);
    itemShowFlag[NORMALS]   = new Q3CheckListItem(itemShowOptions, itemShowFlag[WIREFRAME], "Normals", Q3CheckListItem::CheckBox);

#ifdef SOFA_SMP
    itemShowFlag[PROCESSORCOLOR]   = new Q3CheckListItem(itemShowOptions, itemShowFlag[NORMALS], "Processor Color", Q3CheckListItem::CheckBox);
#endif
    insertItem(itemShowAll);
    itemShowAll->insertItem(itemShowVisual); itemShowAll->setOpen(true);
    itemShowVisual->insertItem(itemShowFlag[VISUALMODELS]);
    itemShowAll->insertItem(itemShowBehavior);
    itemShowBehavior->insertItem(itemShowFlag[BEHAVIORMODELS]);
    itemShowBehavior->insertItem(itemShowFlag[FORCEFIELDS]);
    itemShowBehavior->insertItem(itemShowFlag[INTERACTIONFORCEFIELDS]);
    itemShowAll->insertItem(itemShowCollision);
    itemShowCollision->insertItem(itemShowFlag[COLLISIONMODELS]);
    itemShowCollision->insertItem(itemShowFlag[BOUNDINGCOLLISIONMODELS]);
    itemShowAll->insertItem(itemShowMapping);
    itemShowMapping->insertItem(itemShowFlag[MAPPINGS]);
    itemShowMapping->insertItem(itemShowFlag[MECHANICALMAPPINGS]);

    insertItem(itemShowOptions); itemShowOptions->setOpen(true);
    itemShowOptions->insertItem(itemShowFlag[RENDERING]);
    itemShowOptions->insertItem(itemShowFlag[WIREFRAME]);
    itemShowOptions->insertItem(itemShowFlag[NORMALS]);
#ifdef SOFA_SMP
    itemShowOptions->insertItem(itemShowFlag[PROCESSORCOLOR]);
#endif
    for (int i=0; i<ALLFLAGS; ++i)  mapFlag.insert(std::make_pair(itemShowFlag[i],i));
}



void DisplayFlagWidget::findChildren(Q3CheckListItem *item, std::vector<Q3CheckListItem* > &children)
{
    Q3CheckListItem * child = (Q3CheckListItem * )item->firstChild();
    while(child)
    {
        children.push_back(child);
        findChildren(child,children);
        child = (Q3CheckListItem * )child->nextSibling();
    }
}

void DisplayFlagWidget::contentsMousePressEvent ( QMouseEvent * e )
{

    if ( Q3CheckListItem *item = dynamic_cast<Q3CheckListItem *>(itemAt(contentsToViewport(e->pos()))) )
    {
        std::vector< Q3CheckListItem *> childDepending;
        findChildren(item, childDepending);

        bool value=!item->isOn();
        item->setOn(value);

        if (mapFlag.find(item) != mapFlag.end()) emit change(mapFlag[item],value);
        for (unsigned int idxChild=0; idxChild<childDepending.size(); ++idxChild)
        {
            if (mapFlag.find(childDepending[idxChild]) != mapFlag.end()) emit change(mapFlag[childDepending[idxChild]],value);
        }
        emit clicked();
    }
}

bool DisplayFlagsDataWidget::createWidgets()
{
    flags = new DisplayFlagWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->add(flags);
    connect(flags, SIGNAL(clicked()), this, SLOT(setWidgetDirty()));
    //flags->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
    //flags->setMinimumSize(QSize(50,400));
    setMinimumSize(QSize(50,400));
    //setSizePolicy(QSizePolicy::Preferred,QSizePolicy::Preferred);
    return true;
}

void DisplayFlagsDataWidget::setDataReadOnly(bool readOnly)
{
    flags->setEnabled(!readOnly);
}

void DisplayFlagsDataWidget::readFromData()
{
    const DisplayFlags& displayFlags = this->getData()->getValue();
    if (isRoot)
        flags->setFlag(DisplayFlagWidget::VISUALMODELS, sofa::core::visual::merge_tristate(true,displayFlags.getShowVisualModels()));
    else
        flags->setFlag(DisplayFlagWidget::VISUALMODELS, displayFlags.getShowVisualModels());
    flags->setFlag(DisplayFlagWidget::BEHAVIORMODELS, displayFlags.getShowBehaviorModels());
    flags->setFlag(DisplayFlagWidget::COLLISIONMODELS, displayFlags.getShowCollisionModels());
    flags->setFlag(DisplayFlagWidget::BOUNDINGCOLLISIONMODELS, displayFlags.getShowBoundingCollisionModels());
    flags->setFlag(DisplayFlagWidget::MAPPINGS, displayFlags.getShowMappings());
    flags->setFlag(DisplayFlagWidget::MECHANICALMAPPINGS, displayFlags.getShowMechanicalMappings());
    flags->setFlag(DisplayFlagWidget::FORCEFIELDS, displayFlags.getShowForceFields());
    flags->setFlag(DisplayFlagWidget::INTERACTIONFORCEFIELDS, displayFlags.getShowInteractionForceFields());
    flags->setFlag(DisplayFlagWidget::RENDERING, displayFlags.getShowRendering());
    flags->setFlag(DisplayFlagWidget::WIREFRAME, displayFlags.getShowWireFrame());
    flags->setFlag(DisplayFlagWidget::NORMALS, displayFlags.getShowNormals());
#ifdef SOFA_SMP
    flags->setFlag(DisplayFlagWidget::PROCESSORCOLOR, displayFlags.getShowProcessorColor());
#endif
}

void DisplayFlagsDataWidget::writeToData()
{
    DisplayFlags& displayFlags = *this->getData()->beginEdit();

    displayFlags.setShowVisualModels(flags->getFlag(DisplayFlagWidget::VISUALMODELS));
    displayFlags.setShowBehaviorModels(flags->getFlag(DisplayFlagWidget::BEHAVIORMODELS));
    displayFlags.setShowCollisionModels(flags->getFlag(DisplayFlagWidget::COLLISIONMODELS));
    displayFlags.setShowBoundingCollisionModels(flags->getFlag(DisplayFlagWidget::BOUNDINGCOLLISIONMODELS));
    displayFlags.setShowMappings(flags->getFlag(DisplayFlagWidget::MAPPINGS));
    displayFlags.setShowMechanicalMappings(flags->getFlag(DisplayFlagWidget::MECHANICALMAPPINGS));
    displayFlags.setShowForceFields(flags->getFlag(DisplayFlagWidget::FORCEFIELDS));
    displayFlags.setShowInteractionForceFields(flags->getFlag(DisplayFlagWidget::INTERACTIONFORCEFIELDS));
    displayFlags.setShowRendering(flags->getFlag(DisplayFlagWidget::RENDERING));
    displayFlags.setShowWireFrame(flags->getFlag(DisplayFlagWidget::WIREFRAME));
    displayFlags.setShowNormals(flags->getFlag(DisplayFlagWidget::NORMALS));
#ifdef SOFA_SMP
    displayFlags.setShowProcessorColor(flags->getFlag(DisplayFlagWidget::PROCESSORCOLOR));
#endif
    this->getData()->endEdit();

}




}
}
}

