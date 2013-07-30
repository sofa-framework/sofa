#include <QString>
#include <QWidgetAction>

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
    
    this->createPosition();
    this->createRadius();
    this->createThreshold();
    
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
    select = l_actions[1];
    
    select->setChecked(false);
    disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::ContourImageToolBoxNoTemplated* lp = CITB();
    
    lp->d_ip.setValue(imageposition);
    lp->d_p.setValue(position3D);
    lp->d_axis.setValue(axis);
    lp->d_value.setValue(value.toStdString());
    
    vecX->setValue(round(imageposition.x()));
    vecY->setValue(round(imageposition.y()));
    vecZ->setValue(round(imageposition.z()));
    
    lp->segmentation();
    
    updateGraphs();
}

void ContourImageToolBoxAction::setImageSize(int xsize,int ysize,int zsize)
{
    vecX->setMaximum(xsize);
    vecY->setMaximum(ysize);
    vecZ->setMaximum(zsize);
    
    vecX->setMinimum(0);
    vecY->setMinimum(0);
    vecZ->setMinimum(0);
}


void ContourImageToolBoxAction::selectionPointButtonClick(bool b)
{
    //select = l_actions[0];
    
    if(b)
    {
        //select->setChecked(true);
        connect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    }
    else
    {
        //select->setChecked(false);
        disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
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
        lineH[i]->setVisible(false);
        lineV[i]->setVisible(false);
    }
    
    updateColor();
}


void ContourImageToolBoxAction::updateGraphs()
{
    sofa::defaulttype::Vec3d pos = CITB()->d_ip.getValue();
    
    //QRectF boundaryXY = GraphXY->itemsBoundingRect();
    
//    std::cout << "updateOnGraphs"<<std::endl;
    
    lineH[0]->setVisible(true);
    lineH[0]->setLine(pos.x()-4,pos.y(),pos.x()+4,pos.y());
    
    lineV[0]->setVisible(true);
    lineV[0]->setLine(pos.x(),pos.y()-4,pos.x(),pos.y()+4);
    
    lineH[1]->setVisible(true);
    lineH[1]->setLine(pos.x()-4,pos.z(),pos.x()+4,pos.z());
    
    lineV[1]->setVisible(true);
    lineV[1]->setLine(pos.x(),pos.z()-4,pos.x(),pos.z()+4);
    
    lineH[2]->setVisible(true);
    lineH[2]->setLine(pos.z()-4,pos.y(),pos.z()+4,pos.y());
    
    lineV[2]->setVisible(true);
    lineV[2]->setLine(pos.z(),pos.y()-4,pos.z(),pos.y()+4);
    
}

void ContourImageToolBoxAction::updateColor()
{
    for(int i=0;i<3;i++)
    {
        lineH[i]->setPen(QPen(this->color()));
        lineV[i]->setPen(QPen(this->color()));
    }
}

void ContourImageToolBoxAction::sectionButtonClick()
{
    //std::cout << "ContourImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = CITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChanged(pos2);
}

void ContourImageToolBoxAction::createPosition()
{
    QVBoxLayout *layout2 = new QVBoxLayout();
    QHBoxLayout *layout = new QHBoxLayout();
    vecX = new QSpinBox(); layout->addWidget(vecX);
    vecY = new QSpinBox(); layout->addWidget(vecY);
    vecZ = new QSpinBox(); layout->addWidget(vecZ);
    
    posGroup = new QGroupBox();
    posGroup->setToolTip("position");
    
    layout2->addWidget(new QLabel("position"));
    layout2->addLayout(layout);
    
    posGroup->setLayout(layout2);
    
    QWidgetAction * ac = new QWidgetAction(this);
    ac->setDefaultWidget(posGroup);
    
    //this->l_widgets.append(posGroup);
    this->l_actions.append(ac);
    
    connect(vecX,SIGNAL(editingFinished()),this,SLOT(positionModified()));
    connect(vecY,SIGNAL(editingFinished()),this,SLOT(positionModified()));
    connect(vecZ,SIGNAL(editingFinished()),this,SLOT(positionModified()));
}

void ContourImageToolBoxAction::createRadius()
{
    QVBoxLayout *layout2 = new QVBoxLayout();
     QHBoxLayout *layout = new QHBoxLayout();
     radius= new QSpinBox(); layout->addWidget(radius);
     
     radiusGroup = new QGroupBox();

     radiusGroup->setToolTip("radius");
     
     layout2->addWidget(new QLabel("radius"));
     layout2->addLayout(layout);
     
     radiusGroup->setLayout(layout2);
     
     QWidgetAction * ac = new QWidgetAction(this);
     ac->setDefaultWidget(radiusGroup);
     
     this->l_actions.append(ac);
    
     
     connect(radius,SIGNAL(editingFinished()),this,SLOT(radiusModified()));
}

void ContourImageToolBoxAction::createThreshold()
{
    QVBoxLayout *layout2 = new QVBoxLayout();
     QHBoxLayout *layout = new QHBoxLayout();
     threshold= new QDoubleSpinBox(); layout->addWidget(threshold);
     
     layout2->addWidget(new QLabel("threshold"));
     layout2->addLayout(layout);
     
     thresholdGroup = new QGroupBox();
     thresholdGroup->setLayout(layout2);
     
     QWidgetAction * ac = new QWidgetAction(this);
     ac->setDefaultWidget(thresholdGroup);
     
     this->l_actions.append(ac);
     
     connect(threshold,SIGNAL(editingFinished()),this,SLOT(thresholdModified()));
}


void ContourImageToolBoxAction::positionModified()
{
    std::cout << "positionModified" << std::endl;
    sofa::defaulttype::Vec3d v(vecX->value(),vecY->value(),vecZ->value());
    
    sofa::component::engine::ContourImageToolBoxNoTemplated* lp = CITB();
    
    lp->d_ip.setValue(v);
    
    lp->segmentation();
    
    updateGraphs();
}

void ContourImageToolBoxAction::radiusModified()
{
    std::cout << "radiusModified" << std::endl;
}

void ContourImageToolBoxAction::thresholdModified()
{
    std::cout << "thresholdModified" << std::endl;
}







SOFA_DECL_CLASS(ContourImageToolBoxAction);

//template class SOFA_IMAGE_API ImageToolBox<ImageUS>;



}
}
}
