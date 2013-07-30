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
    
    
    createAxisSelection();
    createListPointWidget();
    
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
    disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
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
        connect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    }
    else
    {
        //select->setChecked(false);
        disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
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

    emit sectionChanged(pos2);
}

void LabelPointsBySectionImageToolBoxAction::createListPointWidget()
{


    TableWidgetForLabelPointBySectionToolBoxAction * tablewidget = new TableWidgetForLabelPointBySectionToolBoxAction(this);
    

    
    this->l_actions.append(tablewidget);
    

    
    connect(tablewidget,SIGNAL(changeSection(int)),this,SLOT(changeSection(int)));
    

}

void LabelPointsBySectionImageToolBoxAction::createAxisSelection()
{
    xyAxis = new QPushButton("XY axis");
    xzAxis = new QPushButton("XZ axis");
    zyAxis = new QPushButton("ZY axis");
    
    QHBoxLayout * l = new QHBoxLayout();
    
    l->addWidget(xyAxis);
    l->addWidget(xzAxis);
    l->addWidget(zyAxis);
    
    xyAxis->setCheckable(true);
    xzAxis->setCheckable(true);
    zyAxis->setCheckable(true);
    
    xyAxis->setChecked(true);
    xzAxis->setChecked(false);
    zyAxis->setChecked(false);
    
    QVBoxLayout * l2=new QVBoxLayout();
    l2->addWidget(new QLabel("select axis"));
    l2->addLayout(l);
    
    QGroupBox *g= new QGroupBox();
    g->setLayout(l2);
    
    QWidgetAction *wa = new QWidgetAction(this);
    wa->setDefaultWidget(g);
    
    this->l_actions.append(wa);
    
    connect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
}

void LabelPointsBySectionImageToolBoxAction::axisChecked(bool b)
{
    QPushButton* button =  qobject_cast<QPushButton*>(sender());
    
    disconnect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    disconnect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    disconnect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    
    if(button)
    {
        if(b==true)
        {
            // if (map !=0) 
            {
                b=false;
                QMessageBox msgbox;
                msgbox.setText("The list of point is not empty.");
                msgbox.setInformativeText("this action will destruct all data.");
                msgbox.setStandardButtons(QMessageBox::Cancel | QMessageBox::Ok);
                msgbox.setDefaultButton(QMessageBox::Cancel);
                msgbox.setIcon(QMessageBox::Warning);
                int ret = msgbox.exec();
                if(ret==QMessageBox::Ok)b=true;
                else
                {
                    button->setChecked(false);
                    connect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
                    connect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
                    connect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
                    return;
                }
            }
        
        }
        if(b==true)
        {
            xyAxis->setChecked(false);
            xzAxis->setChecked(false);
            zyAxis->setChecked(false);
            button->setChecked(true);
        }
        else
        {
            xyAxis->setChecked(false);
            xzAxis->setChecked(false);
            zyAxis->setChecked(false);
            button->setChecked(true);
        }
    }
    connect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    
}

int LabelPointsBySectionImageToolBoxAction::currentAxis()
{
    if(xyAxis->isChecked())return 2;
    if(xzAxis->isChecked())return 1;
    if(zyAxis->isChecked())return 0;
    return -1;
}

void LabelPointsBySectionImageToolBoxAction::changeSection(int i)
{
    sofa::defaulttype::Vec3i v;
    switch(currentAxis())
    {
        case 0:
            v.set(i,0,0);
            break;
        case 1:
            v.set(0,i,0);
            break;
        case 2:
            v.set(0,0,i);
            break;
        default:
            v.set(0,0,0);
    }
    emit sectionChanged(v);
}









SOFA_DECL_CLASS(LabelPointsBySectionImageToolBoxAction)



}
}
}
