#include <QString>
#include <QtGui>

#include "labelpointsbysectionimagetoolboxaction.h"

#include "labelpointsbysectionimagetoolbox.h"

namespace sofa
{
namespace gui
{
namespace qt
{

LabelPointsBySectionImageToolBoxAction::LabelPointsBySectionImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent)
{
    //button selection point
    select = new QAction(this);
    this->l_actions.append(select);
    select->setText("Select Point");
    select->setCheckable(true);
    connect(select,SIGNAL(toggled(bool)),this,SLOT(selectionPointButtonClick(bool)));
    
    QAction* section = new QAction(this);
    this->l_actions.append(section);
    section->setText("Section");
    connect(section,SIGNAL(triggered()),this,SLOT(sectionButtonClick()));
    
}

LabelPointsBySectionImageToolBoxAction::~LabelPointsBySectionImageToolBoxAction()
{
    //delete select;
}


sofa::component::engine::LabelPointsBySectionImageToolBox* LabelPointsBySectionImageToolBoxAction::LPBSITB()
{
    return dynamic_cast<sofa::component::engine::LabelPointsBySectionImageToolBox*>(this->p_label);
}

void LabelPointsBySectionImageToolBoxAction::selectionPointEvent(int /*mouseevent*/, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value)
{
    
    select->setChecked(false);
    disconnect(this,SIGNAL(clickImage(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::LabelPointsBySectionImageToolBox* lp = LPBSITB();
    
    lp->d_ip.setValue(imageposition);
    lp->d_p.setValue(position3D);
    lp->d_axis.setValue(axis);
    lp->d_value.setValue(value.toStdString());
    
    updateGraphs();
}

void LabelPointsBySectionImageToolBoxAction::selectionPointButtonClick(bool b)
{
    
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

void LabelPointsBySectionImageToolBoxAction::addOnGraphs()
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

void LabelPointsBySectionImageToolBoxAction::updateGraphs()
{
    sofa::defaulttype::Vec3d pos = LPBSITB()->d_ip.getValue();
    
    QRectF boundaryXY = GraphXY->itemsBoundingRect();
    
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

void LabelPointsBySectionImageToolBoxAction::updateColor()
{
    for(int i=0;i<3;i++)
    {
        lineH[i]->setPen(QPen(this->color()));
        lineV[i]->setPen(QPen(this->color()));
    }
}

void LabelPointsBySectionImageToolBoxAction::sectionButtonClick()
{
   // std::cout << "LabelPointsBySectionImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = LPBSITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChangeGui(pos2);
}

void LabelPointsBySectionImageToolBoxAction::createListPointWidget()
{
    QComboBox * listSection = new QComboBox();
    QTableWidget * listPoints = new QTableWidget();
    QPushButton * deteteSection  = new QPushButton("del section");
    QPushButton * deletePoint = new QPushButton("del");
    QPushButton * moveupPoint = new QPushButton("up");
    QPushButton * movedownPoint = new QPushButton("down");
    
    QVBoxLayout *vlayout = new QVBoxLayout();
    
    vlayout->addWidget(new QLabel("section"));
    QHBoxLayout *hlayout = new QHBoxLayout();
    hlayout->addWidget(listSection);
    hlayout->addWidget(deteteSection);
    
    vlayout->addLayout(hlayout);
    vlayout->addWidget(listPoints);
    
    QHBoxLayout *hlayout2 = new QHBoxLayout();
    hlayout2->addWidget(deletePoint);
    hlayout2->addWidget(moveupPoint);
    hlayout2->addWidget(movedownPoint);
    
    vlayout->addLayout(hlayout2);
    
    
    QGroupBox *box = new QGroupBox();
    box->setLayout(vlayout);
    
    QWidgetAction *wa = new QWidgetAction(this);
    wa->setDefaultWidget(box);
    
    this->l_actions.append(wa);
}











SOFA_DECL_CLASS(LabelPointsBySectionImageToolBoxAction)



}
}
}
