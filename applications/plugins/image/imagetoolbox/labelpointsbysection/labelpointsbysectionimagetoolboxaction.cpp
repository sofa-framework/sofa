

#include <QString>
#include <QtGui>
#include <QPointF>

#include "labelpointsbysectionimagetoolboxaction.h"

#include "labelpointsbysectionimagetoolbox.h"
#include <sofa/helper/rmath.h>


namespace sofa
{
namespace gui
{
namespace qt
{

LabelPointsBySectionImageToolBoxAction::LabelPointsBySectionImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent),tablewidget(NULL),addPoints(false)
{


    //button selection point

    /*QAction* section = new QAction(this);
    this->l_actions.append(section);
    section->setText("Section");
    connect(section,SIGNAL(triggered()),this,SLOT(sectionButtonClick()));*/

    /*QAction* section = new QAction(this);
        this->l_actions.append(section);
        section->setText("Section");
        connect(section,SIGNAL(triggered()),this,SLOT(sectionButtonClick()));*/
    
    currentSlide = oldSlide = -1;

    createMainCommandWidget();
    createAxisSelectionWidget();
    createListPointWidget();

    reloadData();
    updateData();
    this->tablewidget->updateData();

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
        this->updateGraphs();
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

    if(currentSlide==-1)currentSlide=0;
    changeSection2(currentSlide,true);
}

void LabelPointsBySectionImageToolBoxAction::updateGraphs()
{
    //sofa::defaulttype::Vec3d pos = LPBSITB()->d_ip.getValue();
    //this->updateData();
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

/*void LabelPointsBySectionImageToolBoxAction::sectionButtonClick()
{
   // std::cout << "LabelPointsBySectionImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = LPBSITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChanged(pos2);
}*/

void LabelPointsBySectionImageToolBoxAction::createListPointWidget()
{
    tablewidget = new TableWidgetForLabelPointBySectionToolBoxAction();
    tablewidget->setMapSection(&mapsection);
    
    this->addWidget(tablewidget);
    
    connect(tablewidget,SIGNAL(changeSection(int)),this,SLOT(changeSection(int)));
    

}

void LabelPointsBySectionImageToolBoxAction::createAxisSelectionWidget()
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
    l2->addLayout(l);
    
    QGroupBox *g= new QGroupBox("Axis selection");
    g->setLayout(l2);
    
    this->addWidget(g);

    connect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
}


void LabelPointsBySectionImageToolBoxAction::createMainCommandWidget()
{
    QVBoxLayout *vb=new QVBoxLayout();

    select = new QPushButton("Select Point");
    select->setCheckable(true);
    connect(select,SIGNAL(toggled(bool)),this,SLOT(selectionPointButtonClick(bool)));

    vb->addWidget(select);

    QHBoxLayout *hb=new QHBoxLayout();

    QPushButton *updatePB = new QPushButton("update");
    connect(updatePB,SIGNAL(clicked()),this,SLOT(updateData()));
    QPushButton *reloadPB = new QPushButton("reload");
    connect(reloadPB,SIGNAL(clicked()),this,SLOT(reloadData()));
    QPushButton *loadfilePB = new QPushButton("load file");
    connect(loadfilePB,SIGNAL(clicked()),this,SLOT(loadFileData()));
    QPushButton *savefilePB = new QPushButton("save file");
    connect(savefilePB,SIGNAL(clicked()),this,SLOT(saveFileData()));

    hb->addWidget(updatePB);
    hb->addWidget(reloadPB);
    hb->addWidget(loadfilePB);
    hb->addWidget(savefilePB);

    vb->addLayout(hb);

    QGroupBox *gb = new QGroupBox("Main Commands");

    gb->setLayout(vb);

    this->addWidget(gb);
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

void LabelPointsBySectionImageToolBoxAction::setAxis(int axis)
{
    disconnect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    disconnect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    disconnect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));

    xyAxis->setChecked((axis==2));
    xzAxis->setChecked((axis==1));
    zyAxis->setChecked((axis==0));

    connect(xyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(zyAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
    connect(xzAxis,SIGNAL(toggled(bool)),this,SLOT(axisChecked(bool)));
}

void LabelPointsBySectionImageToolBoxAction::changeSection(int i)
{
    sofa::defaulttype::Vec3i v = changeSection2(i);

    emit sectionChanged(v);
}

sofa::defaulttype::Vec3i LabelPointsBySectionImageToolBoxAction::changeSection2(int i,bool force)
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

    if(i!=currentSlide || force)
    {
        if(mapsection[currentSlide].size()!=0)
        {
            oldpath[currentAxis()]->setPath(path[currentAxis()]->path());
            oldSlide = currentSlide;
        }


        path[0]->setPath(QPainterPath());
        path[1]->setPath(QPainterPath());
        path[2]->setPath(QPainterPath());

        for(unsigned int j=0;j<mapsection[i].size();j++)
        {
            Point &pt = mapsection[i][j];
            addToPath(currentAxis(),pt.ip);
        }

        QMapIterator<unsigned int, VecPointSection> j(mapsection);
        while (j.hasNext())
        {
            j.next();

            unsigned int jj = j.key();
            for(unsigned int k=0;k<mapsection[jj].size();k++)
            {

                Point &pt = mapsection[jj][k];
                addToPath((currentAxis()+1)%3,pt.ip,k==0);
                addToPath((currentAxis()+2)%3,pt.ip,k==0);
            }
        }


        currentSlide = i;
    }

    return v;
}

void LabelPointsBySectionImageToolBoxAction::mouseMove(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition,const sofa::defaulttype::Vec3d& position3D,const QString& )
{
    if(!addPoints)return;
    if((int)axis!=this->currentAxis())return;

    int current_slide = 0;
    switch(axis)
    {
        case 0:
            current_slide = helper::round(imageposition.x());
            break;
        case 1:
            current_slide = helper::round(imageposition.y());
            break;
        case 2:
            current_slide = helper::round(imageposition.z());
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

    addToPath(axis,imageposition);
    addToPath((axis+1)%3,imageposition);
    addToPath((axis+2)%3,imageposition);

    mapsection[current_slide].push_back(p);
    tablewidget->setSection(currentSlide);
}

void LabelPointsBySectionImageToolBoxAction::addToPath(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition, bool forceMoveTo)
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
    if(poly.elementCount() && !forceMoveTo)
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
            this->changeSection2(v.x());
            tablewidget->setSection(helper::round(v.x()));
            break;
        case 1:
            this->changeSection2(v.y());
            tablewidget->setSection(helper::round(v.y()));
            break;
        case 2:
            this->changeSection2(v.z());
            tablewidget->setSection(helper::round(v.z()));
            break;
        default:
            break;
    }
}

void LabelPointsBySectionImageToolBoxAction::updateData()
{
    sofa::component::engine::LabelPointsBySectionImageToolBox *l = LPBSITB();

    int c =currentAxis();

    l->d_axis.setValue(c);

    helper::vector<sofa::defaulttype::Vec3d>& vip = *(l->d_ip.beginEdit());
    helper::vector<sofa::defaulttype::Vec3d>& vp = *(l->d_p.beginEdit());

    //QMapIterator<unsigned int,VecPointSection> i(mapsection);
    QList<unsigned int> list_keys = mapsection.keys();


    vip.clear();
    vp.clear();

    for(int i=0;i<list_keys.size();i++)
    {
        VecPointSection &v = mapsection[list_keys[i]];
        for(unsigned int j=0;j<v.size();j++)
        {
            vip.push_back(v[j].ip);
            vp.push_back(v[j].p);
        }
    }

    l->d_ip.endEdit();
    l->d_p.endEdit();
}

void LabelPointsBySectionImageToolBoxAction::reloadData()
{
    sofa::component::engine::LabelPointsBySectionImageToolBox *l = LPBSITB();
    int axis = l->d_axis.getValue();
    this->setAxis(axis);

    helper::vector<sofa::defaulttype::Vec3d>& vip = *(l->d_ip.beginEdit());
    helper::vector<sofa::defaulttype::Vec3d>& vp = *(l->d_p.beginEdit());

    int size = vip.size();
    if(vip.size() != vp.size())
    {
        std::cerr << "Warning: the imagepositions vector size is different of the 3Dpositions vector size.";
        if(vip.size()>vp.size())size=vp.size();
    }

    mapsection.clear();

    for(int i=0;i<size;i++)
    {
        Point p;
        p.ip = vip[i];
        p.p = vp[i];

        mapsection[helper::round(p.ip[axis])].push_back(p);
    }

    l->d_ip.endEdit();
    l->d_p.endEdit();

    tablewidget->updateData();

}

void LabelPointsBySectionImageToolBoxAction::loadFileData()
{
    sofa::component::engine::LabelPointsBySectionImageToolBox *l = LPBSITB();

    l->loadFile();
    this->setAxis(l->d_axis.getValue());
    this->reloadData();
}

void LabelPointsBySectionImageToolBoxAction::saveFileData()
{
    sofa::component::engine::LabelPointsBySectionImageToolBox *l = LPBSITB();

    this->updateData();
    l->saveFile();
}



SOFA_DECL_CLASS(LabelPointsBySectionImageToolBoxAction)



}
}
}


