#include <QString>
#include <QtGui>
#include <QPointF>

#include "labelpointsbysectionimagetoolboxaction.h"

#include "labelpointsbysectionimagetoolbox.h"



namespace sofa
{
namespace gui
{
namespace qt
{

LabelPointsBySectionImageToolBoxAction::LabelPointsBySectionImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent),addPoints(false),tablewidget(NULL)
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
    
    currentSlide = oldSlide = -1;


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

void LabelPointsBySectionImageToolBoxAction::selectionPointEvent(int mouseevent, const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& value)
{
    if(mouseevent==0)
    {
        this->addPoints=true;
        this->mouseMove(axis,imageposition,position3D,value);
    }
    else
    {
        this->addPoints=false;

        this->tablewidget->updateData();
    }

    
    //updateGraphs();
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


    path[0] = GraphZY->addPath(QPainterPath());
    path[1] = GraphXZ->addPath(QPainterPath());
    path[2] = GraphXY->addPath(QPainterPath());
    
    oldpath[0] = GraphZY->addPath(QPainterPath());
    oldpath[1] = GraphXZ->addPath(QPainterPath());
    oldpath[2] = GraphXY->addPath(QPainterPath());
    
    for(int i=0;i<3;i++)
    {
        path[i]->setVisible(true);
        oldpath[i]->setVisible(true);
    }
    
    updateColor();
}

void LabelPointsBySectionImageToolBoxAction::updateGraphs()
{
    //sofa::defaulttype::Vec3d pos = LPBSITB()->d_ip.getValue();
}

void LabelPointsBySectionImageToolBoxAction::updateColor()
{
    for(int i=0;i<3;i++)
    {
        path[i]->setPen(QPen(this->color()));
        QColor c2 = this->color();
        c2.setAlphaF(c2.alphaF()/3);
        oldpath[i]->setPen(QPen(c2));
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
    tablewidget = new TableWidgetForLabelPointBySectionToolBoxAction(this);
    tablewidget->setMapSection(&mapsection);
    
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
                msgbox.setInformativeText("this action will remove all data.");
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
            mapsection.clear();
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

    std::cout << "changesection"<<std::endl;
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

    if(i!=currentSlide)
    {
        if(mapsection[currentSlide].size()!=0)
        {
            oldpath[currentAxis()]->setPath(path[currentAxis()]->path());
            oldSlide = currentSlide;
        }


        path[currentAxis()]->setPath(QPainterPath());
        for(unsigned int j=0;j<mapsection[i].size();j++)
        {
            Point &pt = mapsection[i][j];
            addToPath(currentAxis(),pt.ip,pt.p);
        }

        currentSlide = i;
    }

    emit sectionChanged(v);
}
/*
void LabelPointsBySectionImageToolBoxAction::changeSection2(int i)
{

    std::cout << "changesection"<<std::endl;
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
}*/

void LabelPointsBySectionImageToolBoxAction::mouseMove(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& )
{
    if(!addPoints)return;
    if(axis!=this->currentAxis())return;

    int current_slide = 0;
    switch(axis)
    {
        case 0:
            current_slide = round(imageposition.x());
            break;
        case 1:
            current_slide = round(imageposition.y());
            break;
        case 2:
            current_slide = round(imageposition.z());
            break;
        default:
            return;
    }
    if(currentSlide!=current_slide)
    {
        changeSection(current_slide);
    }

    Point p;
    p.ip=imageposition;
    p.p=position3D;

    addToPath(axis,imageposition,position3D);

    mapsection[current_slide].push_back(p);
    tablewidget->setSection(currentSlide);
    tablewidget->updateData();
}

void LabelPointsBySectionImageToolBoxAction::addToPath(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& )
{
    QPainterPath poly;
    int ximage,yimage;

    switch(axis)
    {
        case 0:
            ximage = 2; yimage = 1;
            break;
        case 1:
            ximage = 0; yimage = 2;
            break;
        case 2:
            ximage = 0; yimage = 1;
            break;
        default:
            return;
    }

    poly = path[axis]->path();
    if(poly.elementCount())
        poly.lineTo(imageposition[ximage],imageposition[yimage]);
    else
        poly.moveTo(imageposition[ximage],imageposition[yimage]);

    path[axis]->setPath(poly);
}


void LabelPointsBySectionImageToolBoxAction::optionChangeSection(sofa::defaulttype::Vec3i v)
{
    switch (currentAxis())
    {
        case 0:
            this->changeSection(v.x());
            break;
        case 1:
            this->changeSection(v.y());
            break;
        case 2:
            this->changeSection(v.z());
            break;
        default:
            break;
    }
}



SOFA_DECL_CLASS(LabelPointsBySectionImageToolBoxAction)



}
}
}
