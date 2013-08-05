#ifndef IMAGETOOLBOXLABELACTIONWIDGET_H
#define IMAGETOOLBOXLABELACTIONWIDGET_H

#include <QtGui>
#include "imagetoolboxcentralwidget.h"
#include "labelimagetoolboxaction.h"

namespace sofa
{
namespace gui
{
namespace qt
{

class ImageToolBoxLabelActionWidget: public QToolBar
{
Q_OBJECT

    QComboBox *labelSelection;
    
    unsigned int currentAxis;
    Vec3f currentImagePosition;
    Vec3f current3DPosition;
    QString currentVal;
    
    typedef sofa::component::engine::LabelImageToolBox Label;
    typedef sofa::gui::qt::LabelImageToolBoxAction LabelAction;
    typedef helper::vector<Label*> VecLabel;
    typedef helper::vector<LabelAction*> VecLabelAction;
    
    VecLabelAction vecLabelAction;
    int currentLabel;
    
    QGraphicsScene *GraphXY;
    QGraphicsScene *GraphXZ;
    QGraphicsScene *GraphZY;

public:
    ImageToolBoxLabelActionWidget():QToolBar("LabelAction"),
        currentLabel(0),GraphXY(NULL),GraphXZ(NULL),GraphZY(NULL)
    {
        this->setToolTip("LabelAction");
        
        labelSelection = new QComboBox();
        
        this->addWidget(labelSelection);
        
        connect(labelSelection,SIGNAL(currentIndexChanged(int)),this,SLOT(changedLabelIndex(int)));
    }
    
    void connectCentralW(ImageToolBoxCentralWidget* cw)
    {
        connect(cw,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
        connect(cw,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SLOT(setValueOnPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
        
        connect(cw,SIGNAL(mousedoubleclickevent()),this,SLOT(graph_mousedoubleclickevent()));
        connect(cw,SIGNAL(mousepressevent()),this,SLOT(graph_mousepressevent()));
        connect(cw,SIGNAL(mousereleaseevent()),this,SLOT(graph_mousereleaseevent()));
        
        connect(this,SIGNAL(labelChangeGui(sofa::defaulttype::Vec3i)),cw,SLOT(setSliders(sofa::defaulttype::Vec3i)));
        connect(cw,SIGNAL(sliderChanged(sofa::defaulttype::Vec3i)),this,SIGNAL(optionChangeSection(sofa::defaulttype::Vec3i)));
        
    }
    
    void setGraphScene(QGraphicsScene *XY,QGraphicsScene *XZ,QGraphicsScene *ZY)
    {
        GraphXY = XY;
        GraphXZ = XZ;
        GraphZY = ZY;
    }
    
    void clearVecLabelAction()
    {
        
        while(vecLabelAction.size()>0)
        {
            LabelAction *la = vecLabelAction.back();
            
            if(la)
            {
                QList<QAction*> lista = la->getActions();
                disconnect(this,SIGNAL(mouseevent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),la,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
                while(lista.size()>0)
                {
                    this->removeAction(lista.first());
                    lista.removeFirst();
                }
            }
            delete la;
            vecLabelAction.pop_back();
            
        }
    }
    
    void setLabels(const VecLabel &vl)
    {
        VecLabel& vecLabel = const_cast<helper::vector< sofa::component::engine::LabelImageToolBox*>& >(vl);
        
        labelSelection->clear();
        clearVecLabelAction();
        
        
        
        for(unsigned int i=0; i<vecLabel.size();i++)
        {
            Label *v = vecLabel[i];
            //std::cout << "class" <<v->getName()<<std::endl;
            labelSelection->addItem(QString::fromStdString(v->getName()));
            
            LabelAction *la = v->createTBAction(this);
            vecLabelAction.push_back(la);
            this->addActions(la->getActions());
            
            /*for(int j=0;j<la->getWidgets().size();j++)
                this->addWidget(la->getWidgets()[j]);*/
            
            
            connect(this,SIGNAL(mouseevent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),la,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            connect(this,SIGNAL(onPlane(uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),la,SLOT(mouseMove(uint,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            connect(la,SIGNAL(sectionChanged(sofa::defaulttype::Vec3i)),this,SIGNAL(labelChangeGui(sofa::defaulttype::Vec3i)));
            connect(this,SIGNAL(optionChangeSection(sofa::defaulttype::Vec3i)),la,SLOT(optionChangeSection(sofa::defaulttype::Vec3i)));
            
            la->setGraphScene(GraphXY,GraphXZ,GraphZY);
        }
        
        changedLabelIndex(0);
        
    }
    
public slots:
    void setValueOnPlane(unsigned int axis,sofa::defaulttype::Vec3d ip,sofa::defaulttype::Vec3d p ,QString val)
    {
        //std::cout << "testOnPlane: axis=" << axis << " ; ip=" << ip << " ; p="<<p<< " ; val="<<val.toStdString()<<std::endl; 
        
        currentAxis=axis;
        currentImagePosition = ip;
        current3DPosition = p;
        currentVal =val;
    }
    
public slots:
    void graph_mousereleaseevent()
    {
        emit mouseevent(1,currentAxis,currentImagePosition,current3DPosition,currentVal);
    }
    
    void graph_mousepressevent()
    {
        emit mouseevent(0,currentAxis,currentImagePosition,current3DPosition,currentVal);
    }
    
    void graph_mousedoubleclickevent()
    {
        emit mouseevent(2,currentAxis,currentImagePosition,current3DPosition,currentVal);
    }
    
    void changedLabelIndex(int i)
    {
        for(int j=0;j<(int)vecLabelAction.size();j++)
        {
            LabelAction *la = vecLabelAction[j];
            if(j!=i)
            {
                la->setVisible(false);
            }
            else
            {
                la->setVisible(true);
            }
        }
    }
    
signals:
    void mouseevent(int event,const unsigned int axis,const sofa::defaulttype::Vec3d& ip,const sofa::defaulttype::Vec3d& p,const QString& val);

    void onPlane(const unsigned int axis,const sofa::defaulttype::Vec3d& ip,const sofa::defaulttype::Vec3d& p,const QString& val);
    
    void clickOnButton();

    void labelChangeGui(sofa::defaulttype::Vec3i);
    void optionChangeSection(sofa::defaulttype::Vec3i);
};

}
}
}

#endif // IMAGETOOLBOXLABELACTIONWIDGET_H
