#include "labelgridimagetoolboxaction.h"
#include "labelgridimagetoolbox.h"

#include <QFormLayout>
#include <QLabel>


namespace sofa
{
namespace gui
{
namespace qt
{

LabelGridImageToolBoxAction::LabelGridImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
    LabelImageToolBoxAction(lba,parent)
{
 /*   QGroupBox *gb = new QGroupBox();
    gb->setTitle("Main Commands");

    QHBoxLayout *hb = new QHBoxLayout();


    //button selection point
    select = new QPushButton("Select Point");
    hb->addWidget(select);
    //this->addWidget(select);
    select->setCheckable(true);
    connect(select,SIGNAL(toggled(bool)),this,SLOT(selectionPointButtonClick(bool)));

    QPushButton* section = new QPushButton("Go to");
    hb->addWidget(section);
    //this->addWidget(section);
    connect(section,SIGNAL(clicked()),this,SLOT(sectionButtonClick()));

    gb->setLayout(hb);
    this->addWidget(gb);*/


    createMainCommands();
    //createInfoWidget();
    //createBoxCommands();
    createGridCommands();
    //createNormalCommands();

    this->addStretch();

    //executeButtonClick();
}

void LabelGridImageToolBoxAction::createMainCommands()
{
    QGroupBox *gb = new QGroupBox("Main Commands");

    QHBoxLayout *hb = new QHBoxLayout();

//    QPushButton *reloadButton = new QPushButton("Reload");
    //QPushButton *executeButton = new QPushButton("Execute");

//    hb->addWidget(reloadButton);
    //hb->addWidget(executeButton);

    QHBoxLayout *hb2 = new QHBoxLayout();

    QPushButton *saveParamButton = new QPushButton("Save params");
    QPushButton *loadParamButton = new QPushButton("Load params");

    hb2->addWidget(saveParamButton);
    hb2->addWidget(loadParamButton);

    QVBoxLayout *vb =new QVBoxLayout();
    vb->addLayout(hb);
    vb->addLayout(hb2);

    gb->setLayout(vb);

    this->addWidget(gb);
}

/*
void LabelGridImageToolBoxAction::createBoxCommands()
{
    QGroupBox *gb = new QGroupBox("Box Commands");

    QHBoxLayout *hb = new QHBoxLayout();

    QPushButton *cutBoxButton = new QPushButton("Cut");
    QCheckBox *cutBoxCheck = new QCheckBox();

    hb->addWidget(cutBoxButton);
    hb->addWidget(cutBoxCheck);

    gb->setLayout(hb);

    this->addWidget(gb);


}*/

void LabelGridImageToolBoxAction::createGridCommands()
{
    sofa::component::engine::LabelGridImageToolBoxNoTemplated *l = LGITB();

    QGroupBox *gb = new QGroupBox("Grid Commands");

    QVBoxLayout *vb = new QVBoxLayout();
    QHBoxLayout *hb = new QHBoxLayout();

    QPushButton *genGridButton = new QPushButton("Generate Grid");
    //QCheckBox *genGridCheck = new QCheckBox();
    connect(genGridButton,SIGNAL(clicked()),this,SLOT(executeButtonClick()));

    hb->addWidget(genGridButton);
    //hb->addWidget(genGridCheck);

    QFormLayout *fl = new QFormLayout();


    mainAxisSpin = new QSpinBox();
    secondAxisSpin = new QSpinBox();
    mainAxisSpin->setValue(l->d_reso.getValue()[0]);
    secondAxisSpin->setValue(l->d_reso.getValue()[1]);

    //QCheckBox * switchAxisCheck = new QCheckBox();
    QLabel * mainAxisLabel = new QLabel("primary Axis");
    QLabel * secondAxisLabel = new QLabel("secondary Axis");
    //QLabel * switchAxisLabel = new QLabel("switchAxisLabel");


    fl->addRow(mainAxisLabel,mainAxisSpin);
    fl->addRow(secondAxisLabel,secondAxisSpin);
    //fl->addRow(switchAxisLabel,switchAxisCheck);*/


    QString points = "empty",box = "empty";
    if(l->labelpoints) points = l->labelpoints->getName().c_str();
    if(l->labelbox) box = l->labelbox->getName().c_str();

    fl->addRow("Name of PBS:",new QLabel(points));
    fl->addRow("Name of Box:",new QLabel(box));

    vb->addLayout(fl);
    vb->addLayout(hb);
    gb->setLayout(vb);

    this->addWidget(gb);
}

/*
void LabelGridImageToolBoxAction::createNormalCommands()
{
    QGroupBox *gb = new QGroupBox("Normal Commands");

    QHBoxLayout *hb = new QHBoxLayout();

    QPushButton *genNormButton = new QPushButton("Generate Normals");
    QCheckBox *genNormCheck = new QCheckBox();

    hb->addWidget(genNormButton);
    hb->addWidget(genNormCheck);

    gb->setLayout(hb);

    this->addWidget(gb);
}*/


LabelGridImageToolBoxAction::~LabelGridImageToolBoxAction()
{
    //delete select;
}


sofa::component::engine::LabelGridImageToolBoxNoTemplated* LabelGridImageToolBoxAction::LGITB()
{
    return dynamic_cast<sofa::component::engine::LabelGridImageToolBoxNoTemplated*>(this->p_label);
}

void LabelGridImageToolBoxAction::executeButtonClick()
{
    sofa::component::engine::LabelGridImageToolBoxNoTemplated *l = LGITB();

    sofa::component::engine::LabelGridImageToolBoxNoTemplated::Vec2ui reso;
    reso.x() = mainAxisSpin->value();
    reso.y() = secondAxisSpin->value();

    l->d_reso.setValue(reso);

    l->executeAction();
    updateGraphs();

    //std::cout << "exitButton"<<std::endl;
}

void LabelGridImageToolBoxAction::selectionPointEvent(int /*mouseevent*/, const unsigned int /*axis*/,const sofa::defaulttype::Vec3d& /*imageposition*/,const sofa::defaulttype::Vec3d& /*position3D*/,const QString& /*value*/)
{
/*
    select->setChecked(false);
    disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::LabelGridImageToolBox* lp = LGITB();
    
    lp->d_ip.setValue(imageposition);
    lp->d_p.setValue(position3D);
    lp->d_axis.setValue(axis);
    lp->d_value.setValue(value.toStdString());
    
    updateGraphs();*/
}

/*
void LabelGridImageToolBoxAction::selectionPointButtonClick(bool b)
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
    
}*/

void LabelGridImageToolBoxAction::addOnGraphs()
{

//    std::cout << "addOnGraph"<<std::endl;

    path[0] = GraphXY->addPath(QPainterPath());
    path[1] = GraphXZ->addPath(QPainterPath());
    path[2] = GraphZY->addPath(QPainterPath());
    
    
    for(int i=0;i<3;i++)
    {
        path[i]->setVisible(true);
    }
    
    updateColor();
}

void LabelGridImageToolBoxAction::moveTo(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition)
{
    QPainterPath poly;
    int ximage,yimage;

    switch(axis)
    {
        case 0:
            ximage = 0; yimage = 1;
            break;
        case 1:
            ximage = 0; yimage = 2;
            break;
        case 2:
            ximage = 2; yimage = 1;
            break;
        default:
            return;
    }

    poly = path[axis]->path();
    poly.moveTo(imageposition[ximage],imageposition[yimage]);
    path[axis]->setPath(poly);
}

void LabelGridImageToolBoxAction::lineTo(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition)
{
    QPainterPath poly;
    int ximage,yimage;

    switch(axis)
    {
        case 0:
            ximage = 0; yimage = 1;
            break;
        case 1:
            ximage = 0; yimage = 2;
            break;
        case 2:
            ximage =2; yimage = 1;
            break;
        default:
            return;
    }

    poly = path[axis]->path();
    poly.lineTo(imageposition[ximage],imageposition[yimage]);
    path[axis]->setPath(poly);
}

void LabelGridImageToolBoxAction::updateGraphs()
{
    sofa::component::engine::LabelGridImageToolBoxNoTemplated *l = LGITB();
    //int axis = l->d_axis.getValue();
    //this->setAxis(axis);

    helper::vector<sofa::defaulttype::Vec3d>& pos = *(l->d_outImagePosition.beginEdit());
    sofa::component::engine::LabelGridImageToolBoxNoTemplated::Edges& edges = *(l->d_outEdges.beginEdit());

    path[0]->setPath(QPainterPath());
    path[1]->setPath(QPainterPath());
    path[2]->setPath(QPainterPath());

    for(unsigned int i=0;i<edges.size();i++)
    {
        sofa::defaulttype::Vec3d p1 = pos[edges[i][0]];
        sofa::defaulttype::Vec3d p2 = pos[edges[i][1]];
        moveTo(0,p1);
        moveTo(1,p1);
        moveTo(2,p1);
        lineTo(0,p2);
        lineTo(1,p2);
        lineTo(2,p2);
    }

}

void LabelGridImageToolBoxAction::updateColor()
{
    for(int i=0;i<3;i++)
    {
        path[i]->setPen(QPen(this->color()));
    }
}

/*
void LabelGridImageToolBoxAction::sectionButtonClick()
{
   // std::cout << "LabelGridImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = LGITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChanged(pos2);
}*/












SOFA_DECL_CLASS(LabelGridImageToolBoxAction)



}
}
}
