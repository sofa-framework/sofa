#include <QString>
#include <QWidgetAction>

#include "distancezoneimagetoolboxaction.h"

#include "zonegeneratorimagetoolbox.h"

namespace sofa
{
namespace gui
{
namespace qt
{

DistanceZoneImageToolBoxAction::DistanceZoneImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent)
{
    //button selection point
    this->createMainCommands();
/*
    this->createPosition();
    this->createRadius();
    this->createThreshold();
*/
    this->addStretch();
    
}

void DistanceZoneImageToolBoxAction::createMainCommands()
{
    //std::cout << "createMainCommand" << std::endl;
    generate = new QPushButton("Generate");
    generate->setCheckable(true);
    connect(generate,SIGNAL(clicked()),this,SLOT(selectionPointButtonClick()));

    seed = new QLineEdit();
    density = new QLineEdit();

    QHBoxLayout* hb = new QHBoxLayout();
    hb->addWidget(seed);
    hb->addWidget(density);

    QVBoxLayout* vb = new QVBoxLayout();
    vb->addWidget(generate);
    vb->addLayout(hb);

    QGroupBox * gb = new QGroupBox("Main Commands");
    gb->setLayout(vb);

    this->addWidget(gb);

    //std::cout << "end of createMainMenu"<<std::endl;
}

DistanceZoneImageToolBoxAction::~DistanceZoneImageToolBoxAction()
{
    //delete select;
}


sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* DistanceZoneImageToolBoxAction::ZGITB()
{
    return dynamic_cast<sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated*>(this->p_label);
}


void DistanceZoneImageToolBoxAction::selectionPointEvent(int /*mouseevent*/, const unsigned int /*axis*/,const sofa::defaulttype::Vec3d& /*imageposition*/,const sofa::defaulttype::Vec3d& /*position3D*/,const QString& /*value*/)
{
 /*
    select->setChecked(false);
    disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();

    lp->d_ip.setValue(imageposition);
    lp->d_p.setValue(position3D);
    lp->d_axis.setValue(axis);
    lp->d_value.setValue(value.toStdString());

    lp->threshold.setValue(threshold->value());
    lp->radius.setValue(radius->value());
    
    vecX->setValue(round(imageposition.x()));
    vecY->setValue(round(imageposition.y()));
    vecZ->setValue(round(imageposition.z()));
    
    lp->segmentation();
    
    updateGraphs();
    emit updateImage();*/
}

/*void DistanceZoneImageToolBoxAction::setImageSize(int xsize,int ysize,int zsize)
{
    vecX->setMaximum(xsize);
    vecY->setMaximum(ysize);
    vecZ->setMaximum(zsize);
    
    vecX->setMinimum(0);
    vecY->setMinimum(0);
    vecZ->setMinimum(0);
}*/


void DistanceZoneImageToolBoxAction::generatePointButtonClick()
{
    //select = l_actions[0];
    
    ZGITB()->generate();
    
}


void DistanceZoneImageToolBoxAction::addOnGraphs()
{

//    std::cout << "addOnGraph"<<std::endl;
/*
    cursor[0] = GraphXY->addPath(QPainterPath());
    cursor[1] = GraphXZ->addPath(QPainterPath());
    cursor[2] = GraphZY->addPath(QPainterPath());
    
    path[0] = GraphXY->addPath(QPainterPath());
    path[1] = GraphXZ->addPath(QPainterPath());
    path[2] = GraphZY->addPath(QPainterPath());
    
    for(int i=0;i<3;i++)
    {
        path[i]->setVisible(true);
        cursor[i]->setVisible(true);
    }
    
    updateColor();*/
}
/*
QPainterPath DistanceZoneImageToolBoxAction::drawCursor(double x,double y)
{
    QPainterPath c;

    c.moveTo(x-4,y);
    c.lineTo(x+4,y);
    c.moveTo(x,y-4);
    c.lineTo(x,y+4);

    return c;
}

QPainterPath DistanceZoneImageToolBoxAction::drawSegment(VecPixCoord &v, unsigned int axis)
{
    QPainterPath p;

    int idX=0,idY=0;
    switch(axis)
    {
        case 0:
            idX=0;idY=1;
        break;
        case 1:
            idX=0;idY=2;
        break;
        default:
            idX=2;idY=1;
        break;
    }

    for(unsigned int i=0;i<v.size();i++)
    {
        PixCoord &pos= v[i];

        p.addRect(QRectF(pos[idX],pos[idY],1,1));
    }
    return p;
}

void DistanceZoneImageToolBoxAction::drawSegment()
{
    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* l = CITB();


    VecPixCoord& vip = *(l->d_vecPixCoord.beginEdit());

    sofa::defaulttype::Vec3i &pos = sectionPosition;


    VecPixCoord v1,v2,v3;

    for(unsigned int i=0;i<vip.size();i++)
    {
        PixCoord &ip = vip[i];

        if(ip.z()==(unsigned int) pos.z())v1.push_back(ip);
        if(ip.y()==(unsigned int) pos.y())v2.push_back(ip);
        if(ip.x()==(unsigned int) pos.x())v3.push_back(ip);
    }

    path[0]->setPath(drawSegment(v1,0));
    path[1]->setPath(drawSegment(v2,1));
    path[2]->setPath(drawSegment(v3,2));

    l->d_ip.endEdit();
    l->d_p.endEdit();

}*/


void DistanceZoneImageToolBoxAction::updateGraphs()
{
/*    sofa::defaulttype::Vec3d pos = CITB()->d_ip.getValue();
    
    //QRectF boundaryXY = GraphXY->itemsBoundingRect();
    
    //std::cout << "updateOnGraphs"<<std::endl;
    
    cursor[0]->setVisible(true);
    cursor[0]->setPath(drawCursor(pos.x(),pos.y()));

    cursor[1]->setVisible(true);
    cursor[1]->setPath(drawCursor(pos.x(),pos.z()));

    cursor[2]->setVisible(true);
    cursor[2]->setPath(drawCursor(pos.z(),pos.y()));

    drawSegment();

*/
}
/*
void DistanceZoneImageToolBoxAction::updateColor()
{
    for(int i=0;i<3;i++)
    {
        path[i]->setPen(QPen(this->color()));
        path[i]->setBrush(QBrush(this->color()));
        cursor[i]->setPen(QPen(this->color()));
    }
}

void DistanceZoneImageToolBoxAction::sectionButtonClick()
{
    //std::cout << "DistanceZoneImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = CITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChanged(pos2);
}

void DistanceZoneImageToolBoxAction::createPosition()
{

    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();

    QHBoxLayout *layout = new QHBoxLayout();

    vecX = new QSpinBox();
    vecY = new QSpinBox();
    vecZ = new QSpinBox();

    unsigned int x, y, z;
    lp->getImageSize(x, y, z);

    setImageSize(x, y, z);


    sofa::defaulttype::Vec3d pos = lp->d_ip.getValue();
    vecX->setValue(round(pos.x()));
    vecY->setValue(round(pos.y()));
    vecZ->setValue(round(pos.z()));

    layout->addWidget(vecX);
    layout->addWidget(vecY);
    layout->addWidget(vecZ);

    posGroup = new QGroupBox("Position");
    //posGroup->setToolTip("position");

    posGroup->setLayout(layout);
    

    //this->l_widgets.append(posGroup);
    this->addWidget(posGroup);

    connect(vecX,SIGNAL(editingFinished()),this,SLOT(positionModified()));
    connect(vecY,SIGNAL(editingFinished()),this,SLOT(positionModified()));
    connect(vecZ,SIGNAL(editingFinished()),this,SLOT(positionModified()));

}

void DistanceZoneImageToolBoxAction::createRadius()
{
    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();
     QHBoxLayout *layout = new QHBoxLayout();
     radius= new QSpinBox(); layout->addWidget(radius);
     
     radiusGroup = new QGroupBox("Radius");

     //radiusGroup->setToolTip("radius");
     
     radiusGroup->setLayout(layout);

     int rad = lp->radius.getValue();
     radius->setValue(rad);
     
     this->addWidget(radiusGroup);
    
     connect(radius,SIGNAL(editingFinished()),this,SLOT(radiusModified()));
}

void DistanceZoneImageToolBoxAction::createThreshold()
{
    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();
   // QVBoxLayout *layout2 = new QVBoxLayout();
     QHBoxLayout *layout = new QHBoxLayout();
     threshold= new QDoubleSpinBox(); threshold->setSingleStep(0.1); layout->addWidget(threshold);
     
     thresholdGroup = new QGroupBox("Threshold");
     thresholdGroup->setLayout(layout);
     
     int th = lp->threshold.getValue();
     threshold->setValue(th);

     this->addWidget(thresholdGroup);
     
     connect(threshold,SIGNAL(editingFinished()),this,SLOT(thresholdModified()));
}


void DistanceZoneImageToolBoxAction::positionModified()
{
    //std::cout << "positionModified" << std::endl;
    sofa::defaulttype::Vec3d v(vecX->value(),vecY->value(),vecZ->value());
    
    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();
    
    lp->d_ip.setValue(v);
    
    lp->segmentation();
    
    updateGraphs();
    //emit updateImage();
}

void DistanceZoneImageToolBoxAction::radiusModified()
{
    //std::cout << "radiusModified" << std::endl;

    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();

    lp->radius.setValue(radius->value());

    lp->segmentation();

    updateGraphs();
    //emit updateImage();
}

void DistanceZoneImageToolBoxAction::thresholdModified()
{
    //std::cout << "thresholdModified" << std::endl;

    sofa::component::engine::ZoneGeneratorImageToolBoxNoTemplated* lp = CITB();

    lp->threshold.setValue(threshold->value());

    lp->segmentation();

    updateGraphs();
    //emit updateImage();
}
*/
void DistanceZoneImageToolBoxAction::optionChangeSection(sofa::defaulttype::Vec3i /*v*/)
{
  //  sectionPosition = v;

   // updateGraphs();
}






SOFA_DECL_CLASS(DistanceZoneImageToolBoxAction);

//template class SOFA_IMAGE_GUI_API ImageToolBox<ImageUS>;



}
}
}
