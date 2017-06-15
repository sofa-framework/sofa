/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/gui/qt/DisplayFlagsDataWidget.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLayout>

namespace sofa
{
namespace gui
{
namespace qt
{
using namespace sofa::core::objectmodel;
using namespace sofa::core::visual;

helper::Creator<DataWidgetFactory, DisplayFlagsDataWidget > DWClass_DisplayFlags("widget_displayFlags",true);



DisplayFlagWidget::DisplayFlagWidget(QWidget* parent, const char* name,  Qt::WindowFlags f ):
    QTreeWidget(parent /*,name ,f*/)
{
    this->setWindowFlags(f);
    this->setObjectName(name);

    //addColumn(QString::null);
    setRootIsDecorated( true );
    //setTreeStepSize( 12 );
    header()->hide();
    clear();


    setMouseTracking(false);

    //setFocusPolicy(Qt::NoFocus);

    setFrameShadow(QFrame::Plain);
    setFrameShape(QFrame::NoFrame );

    this->setSortingEnabled(false);
    //setSortColumn(-1);
    QTreeWidgetItem* itemShowAll = new QTreeWidgetItem(this);
    this->setTreeWidgetNodeCheckable(itemShowAll, "All");

    QTreeWidgetItem* itemShowVisual    = new QTreeWidgetItem(itemShowAll);
    this->setTreeWidgetNodeCheckable(itemShowVisual, "Visual");

    itemShowFlag[VISUALMODELS]   = new QTreeWidgetItem(itemShowVisual);
    this->setTreeWidgetCheckable(itemShowFlag[VISUALMODELS], "Visual Models");

    QTreeWidgetItem* itemShowBehavior  = new QTreeWidgetItem(itemShowAll);
    this->setTreeWidgetNodeCheckable(itemShowBehavior, "Behavior");

    itemShowFlag[BEHAVIORMODELS]   = new QTreeWidgetItem(itemShowBehavior);
    this->setTreeWidgetCheckable(itemShowFlag[BEHAVIORMODELS], "Behavior Model");
    itemShowFlag[FORCEFIELDS]   = new QTreeWidgetItem(itemShowBehavior, itemShowFlag[BEHAVIORMODELS]);
    this->setTreeWidgetCheckable(itemShowFlag[FORCEFIELDS], "Force Fields");
    itemShowFlag[INTERACTIONFORCEFIELDS]   = new QTreeWidgetItem(itemShowBehavior, itemShowFlag[FORCEFIELDS]);
    this->setTreeWidgetCheckable(itemShowFlag[INTERACTIONFORCEFIELDS], "Interactions");

    QTreeWidgetItem* itemShowCollision = new QTreeWidgetItem(itemShowAll, itemShowBehavior);
    this->setTreeWidgetNodeCheckable(itemShowCollision, "Collision");
    itemShowFlag[COLLISIONMODELS]   = new QTreeWidgetItem(itemShowCollision);
    this->setTreeWidgetCheckable(itemShowFlag[COLLISIONMODELS], "Collision Models");
    itemShowFlag[BOUNDINGCOLLISIONMODELS]   = new QTreeWidgetItem(itemShowCollision, itemShowFlag[COLLISIONMODELS]);
    this->setTreeWidgetCheckable(itemShowFlag[BOUNDINGCOLLISIONMODELS], "Bounding Trees");
    QTreeWidgetItem* itemShowMapping   = new QTreeWidgetItem(itemShowAll, itemShowCollision);
    this->setTreeWidgetNodeCheckable(itemShowMapping, "Mapping");
    itemShowFlag[MAPPINGS]   = new QTreeWidgetItem(itemShowMapping);
    this->setTreeWidgetCheckable(itemShowFlag[MAPPINGS], "Visual Mappings");
    itemShowFlag[MECHANICALMAPPINGS]   = new QTreeWidgetItem(itemShowMapping, itemShowFlag[MAPPINGS]);
    this->setTreeWidgetCheckable(itemShowFlag[MECHANICALMAPPINGS], "Mechanical Mappings");
    QTreeWidgetItem*  itemShowOptions   = new QTreeWidgetItem(this, itemShowAll);
    this->setTreeWidgetNodeCheckable(itemShowOptions, "Options");
    itemShowFlag[RENDERING]   = new QTreeWidgetItem(itemShowOptions);
    this->setTreeWidgetCheckable(itemShowFlag[RENDERING], "Advanced Rendering");
    itemShowFlag[WIREFRAME]   = new QTreeWidgetItem(itemShowOptions);
    this->setTreeWidgetCheckable(itemShowFlag[WIREFRAME], "Wire Frame");
    itemShowFlag[NORMALS]   = new QTreeWidgetItem(itemShowOptions, itemShowFlag[WIREFRAME]);
    this->setTreeWidgetCheckable(itemShowFlag[NORMALS], "Normals");

#ifdef SOFA_SMP
    itemShowFlag[PROCESSORCOLOR]   = new QTreeWidgetItem(itemShowOptions, itemShowFlag[NORMALS]);
    this->setTreeWidgetCheckable(itemShowFlag[PROCESSORCOLOR], "Processor Color");
#endif
    this->addTopLevelItem(itemShowAll);
    itemShowAll->addChild(itemShowVisual); itemShowAll->setExpanded(true);
    itemShowVisual->addChild(itemShowFlag[VISUALMODELS]);
    itemShowAll->addChild(itemShowBehavior);
    itemShowBehavior->addChild(itemShowFlag[BEHAVIORMODELS]);
    itemShowBehavior->addChild(itemShowFlag[FORCEFIELDS]);
    itemShowBehavior->addChild(itemShowFlag[INTERACTIONFORCEFIELDS]);
    itemShowAll->addChild(itemShowCollision);
    itemShowCollision->addChild(itemShowFlag[COLLISIONMODELS]);
    itemShowCollision->addChild(itemShowFlag[BOUNDINGCOLLISIONMODELS]);
    itemShowAll->addChild(itemShowMapping);
    itemShowMapping->addChild(itemShowFlag[MAPPINGS]);
    itemShowMapping->addChild(itemShowFlag[MECHANICALMAPPINGS]);

    this->addTopLevelItem(itemShowOptions); itemShowOptions->setExpanded(true);
    itemShowOptions->addChild(itemShowFlag[RENDERING]);
    itemShowOptions->addChild(itemShowFlag[WIREFRAME]);
    itemShowOptions->addChild(itemShowFlag[NORMALS]);
#ifdef SOFA_SMP
    itemShowOptions->addChild(itemShowFlag[PROCESSORCOLOR]);
#endif
    for (int i=0; i<ALLFLAGS; ++i)  mapFlag.insert(std::make_pair(itemShowFlag[i],i));
}

void DisplayFlagWidget::setTreeWidgetCheckable(QTreeWidgetItem* w, const char* name)
{
    w->setText(0, name);
    w->setExpanded(true);
    w->setFlags(w->flags() | Qt::ItemIsUserCheckable);

}

void DisplayFlagWidget::setTreeWidgetNodeCheckable(QTreeWidgetItem* w, const char* name)
{
    w->setText(0, name);
    w->setExpanded(true);
    w->setFlags(w->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsTristate);

}

void DisplayFlagWidget::findChildren(QTreeWidgetItem *item, std::vector<QTreeWidgetItem *> &children)
{   
    for(int i=0; i<item->childCount() ; i++)
    {
        QTreeWidgetItem * child = (QTreeWidgetItem * )item->child(i);
        children.push_back(child);
        findChildren(child,children);
    }
}

void DisplayFlagWidget::mouseReleaseEvent ( QMouseEvent * e )
{
    //if ( QTreeWidgetItem *item = dynamic_cast<QTreeWidgetItem *>(itemAt(contentsToViewport(e->pos()))) )
    QTreeWidgetItem *item = this->itemAt(e->pos());

    if ( e->button() == Qt::LeftButton && item )
    {
        bool value = !(item->checkState(0) == Qt::Checked);
        item->setCheckState(0, ( (value) ? Qt::Checked : Qt::Unchecked) );

        emit clicked();
    }
}

bool DisplayFlagsDataWidget::createWidgets()
{
    flags = new DisplayFlagWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(flags);
    connect(flags, SIGNAL(clicked()), this, SLOT(setWidgetDirty()));
    //flags->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
    //flags->setMinimumSize(QSize(50,400));
    setMinimumSize(QSize(50,400));
    layout->setContentsMargins(2,2,4,4);
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

