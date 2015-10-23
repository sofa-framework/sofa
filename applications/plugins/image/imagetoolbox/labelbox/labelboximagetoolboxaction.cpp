#include "labelboximagetoolboxaction.h"
#include "labelboximagetoolbox.h"

#include <QString>
#include <QGroupBox>
#include <iostream>

namespace sofa
{
namespace gui
{
namespace qt
{

LabelBoxImageToolBoxAction::LabelBoxImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent)
{
    createMainCommandWidget();

    this->addStretch();
}

void LabelBoxImageToolBoxAction::createMainCommandWidget()
{
    QHBoxLayout *hb = new QHBoxLayout();

    //button selection point
    select = new QPushButton("Select Point");
    select->setCheckable(true);
    connect(select,SIGNAL(toggled(bool)),this,SLOT(selectionPointButtonClick(bool)));

    QPushButton* deletePoints = new QPushButton("Delete points");
    connect(deletePoints,SIGNAL(clicked()),this,SLOT(deleteButtonClick()));

    hb->addWidget(select);
    hb->addWidget(deletePoints);

    QHBoxLayout *hb2 = new QHBoxLayout();

    QPushButton* saveButton = new QPushButton("save");
    connect(saveButton,SIGNAL(clicked()),this,SLOT(saveButtonClick()));

    QPushButton* loadButton = new QPushButton("load");
    connect(loadButton,SIGNAL(clicked()),this,SLOT(loadButtonClick()));

    hb2->addWidget(saveButton);
    hb2->addWidget(loadButton);

    QVBoxLayout *vb = new QVBoxLayout();
    vb->addLayout(hb);
    vb->addLayout(hb2);

    QGroupBox *g = new QGroupBox("Main Commands");
    g->setLayout(vb);

    this->addWidget(g);

}



LabelBoxImageToolBoxAction::~LabelBoxImageToolBoxAction()
{
    //delete select;
}


sofa::component::engine::LabelBoxImageToolBox* LabelBoxImageToolBoxAction::LBITB()
{
    return dynamic_cast<sofa::component::engine::LabelBoxImageToolBox*>(this->p_label);
}

void LabelBoxImageToolBoxAction::selectionPointEvent(int mouseevent, const unsigned int /*axis*/,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& /*value*/)
{
    if(mouseevent != 0)return;
//    select->setChecked(false);
//    disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::LabelBoxImageToolBox* l = LBITB();
    
    helper::vector<sofa::defaulttype::Vec3d>& vip = *(l->d_ip.beginEdit());
    helper::vector<sofa::defaulttype::Vec3d>& vp = *(l->d_p.beginEdit());

    vip.push_back(imageposition);
    vp.push_back(position3D);

    l->d_ip.endEdit();
    l->d_p.endEdit();

    l->calculatebox();
    
    updateGraphs();
}

void LabelBoxImageToolBoxAction::selectionPointButtonClick(bool b)
{
    
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

void LabelBoxImageToolBoxAction::addOnGraphs()
{

//    std::cout << "addOnGraph"<<std::endl;

    path[0] = GraphXY->addPath(QPainterPath());
    path[1] = GraphXZ->addPath(QPainterPath());
    path[2] = GraphZY->addPath(QPainterPath());
    
    for(int i=0;i<4;i++)
    {
        poly[i] = GraphXY->addPolygon(QPolygonF());
        poly[4+i] = GraphXZ->addPolygon(QPolygonF());
        poly[8+i] = GraphZY->addPolygon(QPolygonF());
    }

    rec[0] = GraphXY->addRect(QRect());
    rec[1] = GraphXZ->addRect(QRect());
    rec[2] = GraphZY->addRect(QRect());
    
    for(int i=0;i<3;i++)
    {
        path[i]->setVisible(true);
        rec[i]->setVisible(false);
    }

    for(int i=0;i<12;i++)
    {
        poly[i]->setVisible(false);
    }
    
    updateColor();
}

void LabelBoxImageToolBoxAction::updateGraphs()
{
    validView();

    sofa::component::engine::LabelBoxImageToolBox* l = LBITB();


    helper::vector<sofa::defaulttype::Vec3d>& vip = *(l->d_ip.beginEdit());

    QPainterPath pp[3];

    bool point2 = false;
    if(vip.size()>=2)point2 = true;
    for(unsigned int i=0;i<vip.size();i++)
    {
        sofa::defaulttype::Vec3d &v = vip[i];

        pp[0].moveTo(v.x()-1,v.y());
        pp[0].lineTo(v.x()+1,v.y());
        pp[0].moveTo(v.x(),v.y()-1);
        pp[0].lineTo(v.x(),v.y()+1);

        pp[1].moveTo(v.x()-1,v.z());
        pp[1].lineTo(v.x()+1,v.z());
        pp[1].moveTo(v.x(),v.z()-1);
        pp[1].lineTo(v.x(),v.z()+1);

        pp[2].moveTo(v.z()-1,v.y());
        pp[2].lineTo(v.z()+1,v.y());
        pp[2].moveTo(v.z(),v.y()-1);
        pp[2].lineTo(v.z(),v.y()+1);
    }

    for(int i=0;i<3;i++)
    {
        path[i]->setPath(pp[i]);
    }

    l->d_ip.endEdit();

    const sofa::defaulttype::Vec6d &v = l->d_ipbox.getValue();


    if(point2)
    {
        for(int i=0;i<12;i++)poly[i]->setVisible(true);

        QRectF rect[3];
        rect[0] = GraphXY->sceneRect();
        rect[1] = GraphXZ->sceneRect();
        rect[2] = GraphZY->sceneRect();

        QRectF box[3];
        box[0] = QRectF(QPointF(v[0],v[1]),QPointF(v[3],v[4]));
        box[1] = QRectF(QPointF(v[0],v[2]),QPointF(v[3],v[5]));
        box[2] = QRectF(QPointF(v[2],v[1]),QPointF(v[5],v[4]));

        for(int i=0;i<3;i++)
        {
            rec[i]->setVisible(sectionIsInclude[i]);

            QRectF &r = rect[i];
            QRectF &b = box[i];


            QPolygonF top;
            top.append(r.topLeft());
            top.append(r.topRight());
            top.append(b.topRight());
            top.append(b.topLeft());

            QPolygonF left;
            left.append(r.topLeft());
            left.append(r.bottomLeft());
            left.append(b.bottomLeft());
            left.append(b.topLeft());

            QPolygonF bottom;
            bottom.append(r.bottomLeft());
            bottom.append(r.bottomRight());
            bottom.append(b.bottomRight());
            bottom.append(b.bottomLeft());

            QPolygonF right;
            right.append(r.topRight());
            right.append(r.bottomRight());
            right.append(b.bottomRight());
            right.append(b.topRight());

            rec[i]->setRect(b);

            poly[i*4]->setPolygon(top);
            poly[i*4+1]->setPolygon(left);
            poly[i*4+2]->setPolygon(bottom);
            poly[i*4+3]->setPolygon(right);

        }
    }
    else
    {
        for(int i=0;i<12;i++) poly[i]->setVisible(false);
        for(int i=0;i<3;i++) rec[i]->setVisible(false);
    }
}

void LabelBoxImageToolBoxAction::updateColor()
{
    QColor c = this->color();

    QColor c2 = c;
    c2.setAlphaF(c2.alphaF()/4);


    for(int i=0;i<3;i++)
    {
        path[i]->setPen(QPen(c));
        rec[i]->setPen(QPen(c2));
        rec[i]->setBrush(QBrush(c2));
    }

    for(int i=0;i<12;i++)
    {
        poly[i]->setPen(QPen(c));
        poly[i]->setBrush(QBrush(c2));
    }
}

void LabelBoxImageToolBoxAction::deleteButtonClick()
{
   // std::cout << "LabelBoxImageToolBoxAction::sectionButtonClick()"<<std::endl;

    sofa::component::engine::LabelBoxImageToolBox* l = LBITB();
    helper::vector<sofa::defaulttype::Vec3d>& vip = *(l->d_ip.beginEdit());
    helper::vector<sofa::defaulttype::Vec3d>& vp = *(l->d_ip.beginEdit());

    vip.clear();
    vp.clear();

    l->d_ipbox.setValue(sofa::defaulttype::Vec6d());
    l->d_pbox.setValue(sofa::defaulttype::Vec6d());

    l->d_ip.endEdit();
    l->d_p.endEdit();

    updateGraphs();
}

void LabelBoxImageToolBoxAction::saveButtonClick()
{
    sofa::component::engine::LabelBoxImageToolBox* l = LBITB();
    l->saveFile();
}

void LabelBoxImageToolBoxAction::loadButtonClick()
{
    sofa::component::engine::LabelBoxImageToolBox* l = LBITB();
    l->loadFile();
    updateGraphs();
}


void LabelBoxImageToolBoxAction::optionChangeSection(sofa::defaulttype::Vec3i v)
{
    sectionPosition = v;

    validView();

    updateGraphs();
}

void LabelBoxImageToolBoxAction::validView()
{
    sofa::defaulttype::Vec3i &v = sectionPosition;

    sofa::component::engine::LabelBoxImageToolBox* l = LBITB();
    const sofa::defaulttype::Vec6d box = l->d_ipbox.getValue();

    bool section[3];

    section[0] = !(v.z() >= box[2] && v.z() <= box[5]);
    section[1] = !(v.y() >= box[1] && v.y() <= box[4]);
    section[2] = !(v.x() >= box[0] && v.x() <= box[3]);

    if(section[0] != sectionIsInclude[0] || section[1] != sectionIsInclude[1] || section[2] != sectionIsInclude[2])
    {
        sectionIsInclude[0]=section[0];
        sectionIsInclude[1]=section[1];
        sectionIsInclude[2]=section[2];
    }
}









SOFA_DECL_CLASS(LabelBoxImageToolBoxAction)



}
}
}


