#ifndef IMAGETOOLBOXCENTRALWIDGET_H
#define IMAGETOOLBOXCENTRALWIDGET_H

/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <image/image_gui/config.h>
#include <sofa/gui/qt/DataWidget.h>
#include <sofa/gui/qt/SimpleDataWidget.h>

#include <QLabel>
#include <QImage>
#include <QSlider>
#include <QString>
#include <QGraphicsView>
#include <QGraphicsScene>

#include <image/ImageTypes.h>
#include "imagetoolboxdata.h"
#include "../image_gui/ImagePlaneWidget.h"

#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/rmath.h>

namespace sofa
{
namespace gui
{
namespace qt
{

using namespace helper;
using namespace cimg_library;
using defaulttype::Vec;




class SOFA_IMAGE_GUI_API ImageToolBoxOptionCentralWidget: public QWidget
{
    Q_OBJECT

public:

    ImageToolBoxOptionCentralWidget(ImagePlaneGraphWidget *s, QWidget *parent,const int minimum, const int maximum,const int val, const int height)
        : QWidget(parent) , S(s)
    {
        slider=new QSlider(Qt::Horizontal,this);
        slider->setPageStep ( 1 );
        slider->setRange(minimum,maximum);
        slider->setValue(val);
        slider->setFixedHeight ( height );
        connect(slider, SIGNAL( valueChanged(int) ), this, SLOT( change(int) ) );

        label=new QLabel(this);
        label->setNum(val);
        label->setFixedHeight ( height );

        //toggle = new QCheckBox(this);
        //toggle->setChecked(true);
        //toggle->setFixedHeight ( height );
        //QObject::connect(toggle, SIGNAL( stateChanged(int) ), this, SLOT( togglestateChanged(int) ) );
        //QObject::connect(this,SIGNAL(toggled(bool)),slider,SLOT(setVisible(bool)));
        //QObject::connect(this,SIGNAL(toggled(bool)),label,SLOT(setVisible(bool)));

        QObject::connect(slider,SIGNAL(valueChanged(int)),this,SIGNAL(valueChanged()));

        QHBoxLayout *layout = new QHBoxLayout(this);
        layout->setMargin(0);
        layout->setSpacing(10);
        //layout->add(toggle);
        layout->addWidget(slider);
        layout->addWidget(label);

        this->setFixedHeight ( height );
    }

    //void setChecked(bool val) { toggle->setChecked(val);}

signals:
    //void toggled(bool);
    void valueChanged();
public slots:

    void change(int i)
    {
        S->fromOption(i);
        label->setNum(i);
    }

    /*void togglestateChanged(int i)
    {
        if(i) {emit toggled(true); S->fromOption(slider->value()); }
        else {emit toggled(false); S->fromOption(slider->maximum()+1); }

    }*/

    void changeSlider ( int delta ) { slider->setValue(slider->value() + delta ); }
    void setSlider ( int value ) { slider->setValue(value ); }

    QSlider *getSlider(){return slider;}

protected:
    //QCheckBox* toggle;
    QSlider *slider;
    QLabel *label;
    ImagePlaneGraphWidget *S;
};








class SOFA_IMAGE_GUI_API ImageToolBoxCentralWidget: public QWidget
{
    Q_OBJECT

public slots:
    virtual void handleSliderPolicies()=0; // needed for synchronization of slider visiblity
    virtual void setVisibleXY(bool)=0;
    virtual void setVisibleXZ(bool)=0;
    virtual void setVisibleZY(bool)=0;
    virtual void setVisualModel(bool)=0;

    virtual void setSliders(sofa::defaulttype::Vec3i v)=0;
    virtual void changeSlider()=0;
    
signals:
    void setCheckedXY(bool);
    void setCheckedXZ(bool);
    void setCheckedZY(bool);

    
    void mousedoubleclickevent();
    void mousepressevent();
    void mousereleaseevent();
    
    
    void sliderChanged(sofa::defaulttype::Vec3i v);
    void onPlane(const unsigned in,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&);
};




template<class T>
class TImageToolBoxCentralWidget: public ImageToolBoxCentralWidget
{
    typedef T ImageToolBoxType;
    typedef typename T::ImagePlaneType ImagePlanetype;
    typedef TImagePlaneGraphWidget<ImagePlanetype> Graph;
    typedef ImageToolBoxOptionCentralWidget Options;
    //typedef QSlider Option;
    typedef QVBoxLayout Layout;

    static const unsigned int optionheight=15;

public:

    Options *optionsXY;
    Graph* graphXY;

    Options *optionsXZ;
    Graph* graphXZ;

    Options *optionsZY;
    Graph* graphZY;

    Layout* container_layout;

    ImagePlaneInfoWidget* info;

    //QCheckBox* togglemodels;


    TImageToolBoxCentralWidget()
        : optionsXY(NULL), graphXY(NULL),
          optionsXZ(NULL), graphXZ(NULL),
          optionsZY(NULL), graphZY(NULL),
          container_layout(NULL), info(NULL)
    {}

    bool createLayout(  )
    {
        container_layout = new Layout(this);
        return true;
    }

    bool createWidgets(const ImagePlanetype& d)
    {
        QWidget* parent = this;
    
        //std::cout << "TImageToolBoxCentralWidget::createWidgets" << std::endl;
        info = new ImagePlaneInfoWidget(parent);
        
        QObject::connect(this,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),info,SLOT(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));

        //togglemodels = new QCheckBox(QString("Visual Models"),parent); togglemodels->setChecked(true);

        if(d.getDimensions()[0]>1 && d.getDimensions()[1]>1)
        {
            graphXY = new Graph(parent,2,d);
         //   QObject::connect(togglemodels, SIGNAL( stateChanged(int) ), graphXY, SLOT( togglemodels(int) ) );
         //   QObject::connect(graphXY,SIGNAL(onPlane(const unsigned int,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&)),info,SLOT(onPlane(const unsigned int,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&)));
            
            QObject::connect(graphXY,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            
            if(d.getDimensions()[2]>1)
            {
                optionsXY = new Options(graphXY,parent,0,graphXY->getIndexMax(),graphXY->getIndex(),optionheight);

                QObject::connect(optionsXY,SIGNAL(valueChanged()),this,SLOT(changeSlider()));

            //    QObject::connect(optionsXY,SIGNAL(toggled(bool)),graphXY,SLOT(setVisible(bool)));
                QObject::connect(graphXY,SIGNAL(wheelevent(int)),optionsXY,SLOT(changeSlider(int)));
                if(graphXY->getIndex()>graphXY->getIndexMax()) emit this->setCheckedXY(false);// optionsXY->setChecked(false);
            }
        }

        if(d.getDimensions()[0]>1 && d.getDimensions()[2]>1)
        {
            graphXZ = new Graph(parent,1,d);
         //   QObject::connect(togglemodels, SIGNAL( stateChanged(int) ), graphXZ, SLOT( togglemodels(int) ) );
         //   QObject::connect(graphXZ,SIGNAL(onPlane(const unsigned int,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&)),info,SLOT(onPlane(const unsigned int,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&)));
            
            QObject::connect(graphXZ,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            
            if(d.getDimensions()[1]>1)
            {
                optionsXZ = new Options(graphXZ,parent,0,graphXZ->getIndexMax(),graphXZ->getIndex(),optionheight);
                QObject::connect(optionsXZ,SIGNAL(valueChanged()),this,SLOT(changeSlider()));

           //     QObject::connect(optionsXZ,SIGNAL(toggled(bool)),graphXZ,SLOT(setVisible(bool)));
                QObject::connect(graphXZ,SIGNAL(wheelevent(int)),optionsXZ,SLOT(changeSlider(int)));
                if(graphXZ->getIndex()>graphXZ->getIndexMax()) emit this->setCheckedXZ(false);//optionsXZ->setChecked(false);
            }
        }

        if(d.getDimensions()[1]>1 && d.getDimensions()[2]>1)
        {
            graphZY = new Graph(parent,0,d);
          //  QObject::connect(togglemodels, SIGNAL( stateChanged(int) ), graphZY, SLOT( togglemodels(int) ) );
         //   QObject::connect(graphZY,SIGNAL(onPlane(const unsigned int,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&)),info,SLOT(onPlane(const unsigned int,const sofa::defaulttype::Vec3d&,const sofa::defaulttype::Vec3d&,const QString&)));
            
            QObject::connect(graphZY,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)),this,SIGNAL(onPlane(unsigned int,sofa::defaulttype::Vec3d,sofa::defaulttype::Vec3d,QString)));
            
            if(d.getDimensions()[0]>1)
            {
                optionsZY = new Options(graphZY,parent,0,graphZY->getIndexMax(),graphZY->getIndex(),optionheight);
                QObject::connect(optionsZY,SIGNAL(valueChanged()),this,SLOT(changeSlider()));

                //   QObject::connect(optionsZY,SIGNAL(toggled(bool)),graphZY,SLOT(setVisible(bool)));
                QObject::connect(graphZY,SIGNAL(wheelevent(int)),optionsZY,SLOT(changeSlider(int)));
                if(graphZY->getIndex()>graphZY->getIndexMax()) emit this->setCheckedZY(false);//optionsZY->setChecked(false);
            }
        }

        if(graphXY && graphXZ && graphZY) // synchronize views
        {
            QObject::connect(graphXY->horizontalScrollBar (),SIGNAL(valueChanged (int)),graphXZ->horizontalScrollBar (),SLOT(setValue(int)));
            QObject::connect(graphXY,SIGNAL(roiHorizontalChanged(const qreal,const qreal)),graphXZ,SLOT(setRoiHorizontal(const qreal,const qreal)));
            QObject::connect(graphXY->verticalScrollBar (),SIGNAL(valueChanged (int)),graphZY->verticalScrollBar (),SLOT(setValue(int)));
            QObject::connect(graphXY,SIGNAL(roiVerticalChanged(const qreal,const qreal)),graphZY,SLOT(setRoiVertical(const qreal,const qreal)));
            QObject::connect(graphXY,SIGNAL(cursorChangedX(const qreal)),graphXZ,SLOT(setCursorX(const qreal)));
            QObject::connect(graphXY,SIGNAL(cursorChangedZ(const qreal)),graphXZ,SLOT(setCursorY(const qreal)));
            QObject::connect(graphXY,SIGNAL(cursorChangedY(const qreal)),graphZY,SLOT(setCursorY(const qreal)));
            QObject::connect(graphXY,SIGNAL(cursorChangedZ(const qreal)),graphZY,SLOT(setCursorX(const qreal)));

            QObject::connect(graphXZ->horizontalScrollBar (),SIGNAL(valueChanged (int)),graphXY->horizontalScrollBar (),SLOT(setValue(int)));
            QObject::connect(graphXZ,SIGNAL(roiHorizontalChanged(const qreal,const qreal)),graphXY,SLOT(setRoiHorizontal(const qreal,const qreal)));
            QObject::connect(graphXZ->verticalScrollBar (),SIGNAL(valueChanged (int)),graphZY->horizontalScrollBar (),SLOT(setValue(int)));
            QObject::connect(graphXZ,SIGNAL(roiVerticalChanged(const qreal,const qreal)),graphZY,SLOT(setRoiHorizontal(const qreal,const qreal)));
            QObject::connect(graphXZ,SIGNAL(cursorChangedX(const qreal)),graphXY,SLOT(setCursorX(const qreal)));
            QObject::connect(graphXZ,SIGNAL(cursorChangedZ(const qreal)),graphXY,SLOT(setCursorY(const qreal)));
            QObject::connect(graphXZ,SIGNAL(cursorChangedY(const qreal)),graphZY,SLOT(setCursorX(const qreal)));
            QObject::connect(graphXZ,SIGNAL(cursorChangedZ(const qreal)),graphZY,SLOT(setCursorY(const qreal)));

            QObject::connect(graphZY->horizontalScrollBar (),SIGNAL(valueChanged (int)),graphXZ->verticalScrollBar (),SLOT(setValue(int)));
            QObject::connect(graphZY,SIGNAL(roiHorizontalChanged(const qreal,const qreal)),graphXZ,SLOT(setRoiVertical(const qreal,const qreal)));
            QObject::connect(graphZY->verticalScrollBar (),SIGNAL(valueChanged (int)),graphXY->verticalScrollBar (),SLOT(setValue(int)));
            QObject::connect(graphZY,SIGNAL(roiVerticalChanged(const qreal,const qreal)),graphXY,SLOT(setRoiVertical(const qreal,const qreal)));
            QObject::connect(graphZY,SIGNAL(cursorChangedX(const qreal)),graphXZ,SLOT(setCursorY(const qreal)));
            QObject::connect(graphZY,SIGNAL(cursorChangedZ(const qreal)),graphXZ,SLOT(setCursorX(const qreal)));
            QObject::connect(graphZY,SIGNAL(cursorChangedY(const qreal)),graphXY,SLOT(setCursorY(const qreal)));
            QObject::connect(graphZY,SIGNAL(cursorChangedZ(const qreal)),graphXY,SLOT(setCursorX(const qreal)));

            QObject::connect(graphXY,SIGNAL(roiResized()), this, SLOT(handleSliderPolicies()));
            QObject::connect(graphXZ,SIGNAL(roiResized()), this, SLOT(handleSliderPolicies()));
            QObject::connect(graphZY,SIGNAL(roiResized()), this, SLOT(handleSliderPolicies()));
            
            
            QObject::connect(graphXY,SIGNAL(mousedoubleclickevent()),this,SIGNAL(mousedoubleclickevent()));
            QObject::connect(graphXY,SIGNAL(mousepressevent()),this,SIGNAL(mousepressevent()));
            QObject::connect(graphXY,SIGNAL(mousereleaseevent()),this,SIGNAL(mousereleaseevent()));
            
            QObject::connect(graphXZ,SIGNAL(mousedoubleclickevent()),this,SIGNAL(mousedoubleclickevent()));
            QObject::connect(graphXZ,SIGNAL(mousepressevent()),this,SIGNAL(mousepressevent()));
            QObject::connect(graphXZ,SIGNAL(mousereleaseevent()),this,SIGNAL(mousereleaseevent()));
            
            QObject::connect(graphZY,SIGNAL(mousedoubleclickevent()),this,SIGNAL(mousedoubleclickevent()));
            QObject::connect(graphZY,SIGNAL(mousepressevent()),this,SIGNAL(mousepressevent()));
            QObject::connect(graphZY,SIGNAL(mousereleaseevent()),this,SIGNAL(mousereleaseevent()));
        }

       // std::cout << "~TImageToolBoxCentralWidget::createWidgets" << std::endl;
        return true;
    }
    
    
       void setReadOnly(bool /*readOnly*/) { }
    void readFromData(const ImagePlanetype& d0) {  if(graphXY) graphXY->readFromData(d0); if(graphXZ) graphXZ->readFromData(d0); if(graphZY) graphZY->readFromData(d0);}
    void writeToData(ImagePlanetype& d) { if(graphXY) graphXY->writeToData(d); if(graphXZ) graphXZ->writeToData(d); if(graphZY) graphZY->writeToData(d);}

    void insertWidgets()
    {
        this->setLayout(container_layout);

        QGridLayout* layout = new QGridLayout();
        if(graphXY) layout->addWidget(graphXY,0,0);
        if(optionsXY) layout->addWidget(optionsXY,1,0);

        if(graphXZ) layout->addWidget(graphXZ,2,0);
        if(optionsXZ) layout->addWidget(optionsXZ,3,0);

        if(graphZY) layout->addWidget(graphZY,0,1);
        if(optionsZY) layout->addWidget(optionsZY,1,1);
        container_layout->addLayout(layout);
       // container_layout->add(togglemodels);

        //if(graphXY && graphXZ && graphZY) layout->addWidget(info,2,1);
        //else
        if(info)container_layout->addWidget(info);

    }

    void handleSliderPolicies()
    {
        if(graphXY && graphXZ && graphZY)
        {
            if(this->graphXY->isRoiResized() || this->graphXZ->isRoiResized() || this->graphZY->isRoiResized())
            {
                this->graphXY->setScrollBarPolicies(true);
                this->graphXZ->setScrollBarPolicies(true);
                this->graphZY->setScrollBarPolicies(true);
            }
            else
            {
                this->graphXY->setScrollBarPolicies(false);
                this->graphXZ->setScrollBarPolicies(false);
                this->graphZY->setScrollBarPolicies(false);
            }
        }
    }
    
    void setVisibleXY(bool v)
    {
        this->graphXY->setVisible(v);
        this->optionsXY->setVisible(v);
    }

    void setVisibleXZ(bool v)
    {
        this->graphXZ->setVisible(v);
        this->optionsXZ->setVisible(v);
    }
    
    void setVisibleZY(bool v)
    {
        this->graphZY->setVisible(v);
        this->optionsZY->setVisible(v);
    }
    
    void setVisualModel(bool v)
    {
        int val = (v)?2:0;
        
        this->graphXY->togglemodels(val);
        this->graphZY->togglemodels(val);
        this->graphXZ->togglemodels(val);
    }
    
    void setSliders(sofa::defaulttype::Vec3i v)
    {
        this->optionsXY->setSlider(v.z());
        this->optionsXZ->setSlider(v.y());
        this->optionsZY->setSlider(v.x());
    }

    void changeSlider()
    {
        sofa::defaulttype::Vec3i v;

        v.set(this->optionsZY->getSlider()->value(),this->optionsXZ->getSlider()->value(),this->optionsXY->getSlider()->value());

        emit sliderChanged(v);
    }
};

}}}


#endif // IMAGETOOLBOXCENTRALWIDGET_H
