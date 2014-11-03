#ifndef IMAGETOOLBOXLABELACTIONWIDGET_H
#define IMAGETOOLBOXLABELACTIONWIDGET_H

#include <QtGui>
#include "imagetoolboxcentralwidget.h"
#include "labelimagetoolboxaction.h"
#include "initImage_gui.h"

namespace sofa
{
namespace gui
{
namespace qt
{

class SOFA_IMAGE_GUI_API ImageToolBoxLabelActionWidget: public QWidget
{
Q_OBJECT

    QVBoxLayout *vlayout;
    QComboBox *labelSelection;
    QPushButton *labelColor;
    QStackedWidget *stack;
    
    unsigned int currentAxis;
    defaulttype::Vec3f currentImagePosition;
    defaulttype::Vec3f current3DPosition;
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
    ImageToolBoxLabelActionWidget():QWidget(),
        currentLabel(0),GraphXY(NULL),GraphXZ(NULL),GraphZY(NULL)
    {
        this->setToolTip("LabelAction");

        labelSelection = new QComboBox();
        labelColor = new QPushButton();
        labelColor->setMinimumSize(30,30);
        labelColor->setMaximumSize(30,30);
        labelColor->setIconSize(QSize(20,20));

        connect(labelColor,SIGNAL(clicked()),this,SLOT(changeColor()));

        QHBoxLayout *hb = new QHBoxLayout();

        hb->addWidget(labelSelection);
        hb->addWidget(labelColor);

        stack = new QStackedWidget;
        stack->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        vlayout = new QVBoxLayout();
        vlayout->addLayout(hb);
        vlayout->addWidget(stack);
        //vlayout->addWidget(new QPushButton("Debug"));

        this->setLayout(vlayout);
        
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
        while(stack->widget(0))
            stack->removeWidget(stack->widget(0));

        while(vecLabelAction.size()>0)
        {
            LabelAction *la = vecLabelAction.back();

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

            QWidget *w = new QWidget();
            w->setLayout(la->layout());

            stack->addWidget(w);
            
            connect(this,SIGNAL(mouseevent(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),la,SIGNAL(clickImage(int,unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            connect(this,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),la,SLOT(mouseMove(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            connect(la,SIGNAL(sectionChanged(sofa::defaulttype::Vec3i)),this,SIGNAL(labelChangeGui(sofa::defaulttype::Vec3i)));
            connect(this,SIGNAL(optionChangeSection(sofa::defaulttype::Vec3i)),la,SLOT(optionChangeSection(sofa::defaulttype::Vec3i)));
            //connect(la,SIGNAL(updateImage()),this,SIGNAL(updateImage()));

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
    
    void changeColor()
    {
        int i=stack->currentIndex();
        if(i!=-1)
        {
            connect(vecLabelAction[i],SIGNAL(colorChanged()),this,SLOT(colorIsChanged()));
            vecLabelAction[i]->clickColor();
        }
    }

    void colorIsChanged()
    {
        LabelImageToolBoxAction *b = qobject_cast<LabelImageToolBoxAction*>(sender());
        if(b)
        {
            disconnect(b,SIGNAL(colorChanged()),this,SLOT(colorIsChanged()));

            setColor();
        }
    }

    void changedLabelIndex(int i)
    {
        stack->setCurrentIndex(i);

        setColor();
    }


    void setColor()
    {
        int i=stack->currentIndex();
        if(i!=-1)
        {
            QColor c = vecLabelAction[i]->color();

            setColor(c);
        }
        else
        {
            QColor c(0,0,0,0);

            setColor(c);
        }
    }

    void setColor(QColor c)
    {
        QPixmap pix(30,30);
        pix.fill(c);
        QIcon icon(pix);
        labelColor->setIcon(icon);
    }
    
signals:
    void mouseevent(int event,const unsigned int axis,const sofa::defaulttype::Vec3d& ip,const sofa::defaulttype::Vec3d& p,const QString& val);

    void onPlane(const unsigned int axis,const sofa::defaulttype::Vec3d& ip,const sofa::defaulttype::Vec3d& p,const QString& val);
    
    void clickOnButton();

    void labelChangeGui(sofa::defaulttype::Vec3i);
    void optionChangeSection(sofa::defaulttype::Vec3i);

    //void updateImage();
};

}
}
}

#endif // IMAGETOOLBOXLABELACTIONWIDGET_H
