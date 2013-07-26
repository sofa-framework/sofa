#include <QString>

#include "contourimagetoolboxaction.h"

#include "contourimagetoolbox.h"

namespace sofa
{
namespace gui
{
namespace qt
{

ContourImageToolBoxAction::ContourImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent)
{
    //button selection point
    QAction* select = new QAction(this);
    this->l_actions.append(select);
    select->setText("Select Point");
    select->setCheckable(true);
    connect(select,SIGNAL(toggled(bool)),this,SLOT(selectionPointButtonClick(bool)));
    
    QAction* section = new QAction(this);
    this->l_actions.append(section);
    section->setText("Section");
    connect(section,SIGNAL(triggered()),this,SLOT(sectionButtonClick()));
    
}


ContourImageToolBoxAction::~ContourImageToolBoxAction()
{
    //delete select;
}


sofa::component::engine::ContourImageToolBoxNoTemplated* ContourImageToolBoxAction::CITB()
{
    return dynamic_cast<sofa::component::engine::ContourImageToolBoxNoTemplated*>(this->p_label);
}


void ContourImageToolBoxAction::selectionPointEvent(int /*mouseevent*/, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value)
{
    QAction* select = l_actions[0];
    
    select->setChecked(false);
    disconnect(this,SIGNAL(clickImage(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::ContourImageToolBoxNoTemplated* lp = CITB();
    
    lp->d_ip.setValue(imageposition);
    lp->d_p.setValue(position3D);
    lp->d_axis.setValue(axis);
    lp->d_value.setValue(value.toStdString());
    
    lp->segmentation();
    
    updateGraphs();
}


void ContourImageToolBoxAction::selectionPointButtonClick(bool b)
{
    //QAction* select = l_actions[0];
    
    if(b)
    {
        //select->setChecked(true);
        connect(this,SIGNAL(clickImage(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    }
    else
    {
        //select->setChecked(false);
        disconnect(this,SIGNAL(clickImage(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    }
    
}


void ContourImageToolBoxAction::addOnGraphs()
{

//    std::cout << "addOnGraph"<<std::endl;

    lineH[0] = GraphXY->addLine(0,0,0,0);
    lineH[1] = GraphXZ->addLine(0,0,0,0);
    lineH[2] = GraphZY->addLine(0,0,0,0);
    
    lineV[0] = GraphXY->addLine(0,0,0,0);
    lineV[1] = GraphXZ->addLine(0,0,0,0);
    lineV[2] = GraphZY->addLine(0,0,0,0);
    
    for(int i=0;i<3;i++)
    {
        lineH[i]->setPen(QPen(QColor(255,0,0)));
        lineV[i]->setPen(QPen(QColor(255,0,0)));
        lineH[i]->setVisible(false);
        lineV[i]->setVisible(false);
    }
}


void ContourImageToolBoxAction::updateGraphs()
{
    sofa::defaulttype::Vec3d pos = CITB()->d_ip.getValue();
    
    //QRectF boundaryXY = GraphXY->itemsBoundingRect();
    
//    std::cout << "updateOnGraphs"<<std::endl;
    
    lineH[0]->setVisible(true);
    lineH[0]->setLine(pos.x()-2,pos.y(),pos.x()+2,pos.y());
    
    lineV[0]->setVisible(true);
    lineV[0]->setLine(pos.x(),pos.y()-2,pos.x(),pos.y()+2);
    
    
    lineH[1]->setVisible(true);
    lineH[1]->setLine(pos.x()-2,pos.z(),pos.x()+2,pos.z());
    
    lineV[1]->setVisible(true);
    lineV[1]->setLine(pos.x(),pos.z()-2,pos.x(),pos.z()+2);
    
    
    lineH[2]->setVisible(true);
    lineH[2]->setLine(pos.z()-2,pos.y(),pos.z()+2,pos.y());
    
    lineV[2]->setVisible(true);
    lineV[2]->setLine(pos.z(),pos.y()-2,pos.z(),pos.y()+2);
    
}


void ContourImageToolBoxAction::sectionButtonClick()
{
    //std::cout << "ContourImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = CITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChangeGui(pos2);
}












SOFA_DECL_CLASS(ContourImageToolBoxAction);

//template class SOFA_IMAGE_API ImageToolBox<ImageUS>;



}
}
}
