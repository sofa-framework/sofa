#include <QString>

#include "depthimagetoolboxaction.h"

#include "depthimagetoolbox.h"

namespace sofa
{
namespace gui
{
namespace qt
{

int DepthRowImageToolBoxAction::numindex=0;

DepthImageToolBoxAction::DepthImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent):
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
    //createGridCommands();
    //createNormalCommands();
    createListLayers();

    //executeButtonClick();
}

void DepthImageToolBoxAction::createMainCommands()
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
    QPushButton *saveSceneButton = new QPushButton("Save SCN");

    connect(saveParamButton,SIGNAL(clicked()),this,SLOT(saveButtonClick()));
    connect(loadParamButton,SIGNAL(clicked()),this,SLOT(loadButtonClick()));
    connect(saveSceneButton,SIGNAL(clicked()),this,SLOT(saveSceneButtonClick()));

    hb2->addWidget(saveParamButton);
    hb2->addWidget(loadParamButton);
    QHBoxLayout *hb3 = new QHBoxLayout();
    hb3->addWidget(saveSceneButton);

    QVBoxLayout *vb =new QVBoxLayout();
    vb->addLayout(hb);
    vb->addLayout(hb2);
    vb->addLayout(hb3);

    gb->setLayout(vb);

    this->addWidget(gb);
}

void DepthImageToolBoxAction::createListLayers()
{
    std::cout<<"nnn "<<std::endl;

    sofa::component::engine::DepthImageToolBox *l = DITB();
    unsigned int numberGrid = l->labelsOfGrid.size();

    QGroupBox * gp = new QGroupBox("List of Layers");

    QFormLayout *form = new QFormLayout();
    form->addRow("numbers of Grids:",new QLabel(QString::number(numberGrid)));

    listLayers = new QTableWidget();

    listLayers->insertColumn(0);
    listLayers->insertColumn(1);
    listLayers->insertColumn(2);
    listLayers->insertColumn(3);

    QStringList header; header << "name" << "geometry" << "grid1" << "grid2";
    listLayers->setHorizontalHeaderLabels(header);


    QPushButton *listLayersAdd = new QPushButton("Add");
    connect(listLayersAdd,SIGNAL(clicked()),this,SLOT(createNewRow()));
    QPushButton *listLayersRem = new QPushButton("remove");

    QPushButton *listLayersGenerate = new QPushButton("generate");
    connect(listLayersGenerate,SIGNAL(clicked()),this,SLOT(executeButtonClick()));

    QHBoxLayout * hb = new QHBoxLayout();
    hb->addWidget(listLayersAdd);
    hb->addWidget(listLayersRem);

    QVBoxLayout *vb = new QVBoxLayout();
    vb->addLayout(form);
    vb->addWidget(listLayers);
    vb->addLayout(hb);
    vb->addWidget(listLayersGenerate);

    gp->setLayout(vb);

    if(numberGrid==0)
    {
        listLayersAdd->setEnabled(false);
        listLayersRem->setEnabled(false);
        //listLayers->setEnabled(false);
    }



    this->addWidget(gp);

}

void DepthImageToolBoxAction::createNewRow(int layer)
{
    sofa::component::engine::DepthImageToolBox *d = DITB();

    sofa::component::engine::DepthImageToolBox::Layer &l = d->layers[layer];

    this->createNewRow();
    DepthRowImageToolBoxAction *row = listRows.back();

    QString offset1;offsetToText(offset1,l.offset1,l.typeOffset1);
    QString offset2;offsetToText(offset2,l.offset2,l.typeOffset2);

    row->setValue(QString::fromStdString(l.name), l.layer1, offset1, l.layer2, offset2, l.base,l.nbSlice);
}

void DepthImageToolBoxAction::createNewRow()
{
    //std::cout << "createNewRow"<<std::endl;

    sofa::component::engine::DepthImageToolBox *l = DITB();
    
    DepthRowImageToolBoxAction *row = new DepthRowImageToolBoxAction(listRows.size());
    row->setParameters(l->labelsOfGrid);

    row->toTableWidgetRow(listLayers);
    listLayers->update();

    l->createLayer();
    //std::cout << l->layers.size()<<std::endl;

    this->connect(row,SIGNAL(valueChanged(int,QString,int,QString,int,QString,int,int)),this,SLOT(changeRow(int,QString,int,QString,int,QString,int,int)));
    row->change();

    listRows.push_back(row);
}

void DepthImageToolBoxAction::offsetToText(QString &out, double outValue, int type)
{
    switch(type)
    {
        case DepthImageToolBox::Layer::Distance:
            out = QString::number(outValue);
            break;
        case DepthImageToolBox::Layer::Percent:
            out = QString::number(outValue*100) +"%";
            break;
    }

    return;
}

void DepthImageToolBoxAction::textToOffset(QString text, double &outValue, int &type)
{
    bool ok;
    outValue = text.toDouble(&ok);
    if(ok)
    {
        type = sofa::component::engine::DepthImageToolBox::Layer::Distance;
        return;
    }

    if(text.contains("%"))
    {
        text.replace("%","");
        outValue = text.toDouble(&ok);
        outValue *= 0.01;
        if(ok)
        {
            type = sofa::component::engine::DepthImageToolBox::Layer::Percent;
            return;
        }
    }

    outValue = 0;
    type = sofa::component::engine::DepthImageToolBox::Layer::Distance;
    return;
}

void DepthImageToolBoxAction::changeRow(int index,QString name,int layer1,QString offset1,int layer2,QString offset2,int base,int nbSlice)
{
    //std::cout << "DepthImageToolBoxAction::changeRow " << index << " "<< name.toStdString() << " "<<layer1<<" "<<offset1.toStdString()<<" "<<layer2<<" "<<offset2.toStdString()<<std::endl;

    sofa::component::engine::DepthImageToolBox *l = DITB();

    if(l->layers.size()<=(unsigned int)index)return;

    sofa::component::engine::DepthImageToolBox::Layer &layer = l->layers[index];

    layer.name = name.toStdString();
    layer.layer1 = layer1;
    layer.layer2 = layer2;
    layer.base = base;
    layer.nbSlice = nbSlice;

    textToOffset(offset1,layer.offset1,layer.typeOffset1);
    textToOffset(offset2,layer.offset2,layer.typeOffset2);

    //std::cout << layer.name << " " << layer.offset1 << " " << layer.typeOffset1 <<layer.offset2 << " " << layer.typeOffset2 <<std::endl;
}

DepthImageToolBoxAction::~DepthImageToolBoxAction()
{
    //delete select;
}

sofa::component::engine::DepthImageToolBox* DepthImageToolBoxAction::DITB()
{
    return dynamic_cast<sofa::component::engine::DepthImageToolBox*>(this->p_label);
}

void DepthImageToolBoxAction::executeButtonClick()
{
    sofa::component::engine::DepthImageToolBox *l = DITB();
    l->executeAction();
    //updateGraphs();

    //std::cout << "exitButton"<<std::endl;
}

void DepthImageToolBoxAction::saveButtonClick()
{
    //std::cout << "saveButtonClick"<<std::endl;
    sofa::component::engine::DepthImageToolBox *l = DITB();
    l->saveFile();
    //updateGraphs();

    //std::cout << "exitButton"<<std::endl;
}

void DepthImageToolBoxAction::loaduttonClick()
{
    sofa::component::engine::DepthImageToolBox *l = DITB();
    l->loadFile();
    updateGraphs();

    //std::cout << "exitButton"<<std::endl;
}

void DepthImageToolBoxAction::saveSceneButtonClick()
{
    sofa::component::engine::DepthImageToolBox *l = DITB();
    l->saveSCN();
}

void DepthImageToolBoxAction::selectionPointEvent(int /*mouseevent*/, const unsigned int /*axis*/,const sofa::defaulttype::Vec3d& /*imageposition*/,const sofa::defaulttype::Vec3d& /*position3D*/,const QString& /*value*/)
{
/*
    select->setChecked(false);
    disconnect(this,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(selectionPointEvent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
    
    sofa::component::engine::DepthImageToolBox* lp = LGITB();
    
    lp->d_ip.setValue(imageposition);
    lp->d_p.setValue(position3D);
    lp->d_axis.setValue(axis);
    lp->d_value.setValue(value.toStdString());
    
    updateGraphs();*/
}

/*
void DepthImageToolBoxAction::selectionPointButtonClick(bool b)
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

void DepthImageToolBoxAction::addOnGraphs()
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

void DepthImageToolBoxAction::moveTo(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition)
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

void DepthImageToolBoxAction::lineTo(const unsigned int axis,const sofa::defaulttype::Vec3d& imageposition)
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

void DepthImageToolBoxAction::updateGraphs()
{
    sofa::component::engine::DepthImageToolBox *l = DITB();

    listLayers->clear();
    listRows.clear();

    for(unsigned int i=0;i<l->layers.size();i++)
    {
        this->createNewRow(i);
    }


    //int axis = l->d_axis.getValue();
    //this->setAxis(axis);

    /*helper::vector<sofa::defaulttype::Vec3d>& pos = *(l->d_outImagePosition.beginEdit());
    sofa::component::engine::DepthImageToolBox::Edges& edges = *(l->d_outEdges.beginEdit());

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
    }*/

}

void DepthImageToolBoxAction::updateColor()
{
    for(int i=0;i<3;i++)
    {
        path[i]->setPen(QPen(this->color()));
    }
}

/*
void DepthImageToolBoxAction::sectionButtonClick()
{
   // std::cout << "DepthImageToolBoxAction::sectionButtonClick()"<<std::endl;
    sofa::defaulttype::Vec3d pos = LGITB()->d_ip.getValue();
    
    sofa::defaulttype::Vec3i pos2(round(pos.x()),round(pos.y()),round(pos.z()));

    emit sectionChanged(pos2);
}*/












SOFA_DECL_CLASS(DepthImageToolBoxAction)



}
}
}
