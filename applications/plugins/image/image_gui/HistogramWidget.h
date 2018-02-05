/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_IMAGE_HISTOGRAMWIDGET_H
#define SOFA_IMAGE_HISTOGRAMWIDGET_H

#include <image/image_gui/config.h>
#include <sofa/gui/qt/DataWidget.h>
#include <sofa/gui/qt/SimpleDataWidget.h>

#include <QLabel>
#include <QImage>
#include <QSlider>
#include <QString>
#include <QGraphicsView>
#include <QGraphicsScene>


#include "../ImageTypes.h"
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

//-----------------------------------------------------------------------------------------------//
//  data widget -> generate qimage from Data<histogram> and show info (min,max,mouse position)
//-----------------------------------------------------------------------------------------------//


class HistogramSetting: public QObject
{
    Q_OBJECT;

public:
    virtual ~HistogramSetting() {};
    virtual void fromOption(const unsigned int i)=0;
    virtual void fromGraph(const QPointF &pt,const bool clicked)=0;
    QImage *getImage() {return &img;}
    QWidget *getWidget() {return widget;}

signals:
    void clampModified();

protected:
    QImage img;	// QImage to draw
    QWidget* widget;
};


template <class DataType>
class THistogramSetting: public HistogramSetting
{
public:
    typedef DataType Histotype;
    typedef typename Histotype::T T;

    THistogramSetting(QWidget * parent )	:histo(NULL),channel(0),channelmax(0),currentpos(0),clamp(defaulttype::Vec<2,T>(0,0))
    {
        label1=new QLabel(parent);
        label2=new QLabel(parent);
        label3=new QLabel(parent);

        QHBoxLayout *layout = new QHBoxLayout(parent);
        layout->setMargin(0);
        layout->addWidget(label1);
        layout->addStretch();
        layout->addWidget(label2);
        layout->addStretch();
        layout->addWidget(label3);
        widget = new QWidget(parent);
        widget->setLayout(layout);
    }

    virtual ~THistogramSetting() {}

    void readFromData(const Histotype& d0)
    {
        this->histo=&d0;
        if(!this->histo) return;
        this->cimg=cimg_library::CImg<unsigned char>(3,this->histo->getImage().width(),this->histo->getImage().height());
        this->img=QImage(cimg.data(),cimg.height(),cimg.depth(),QImage::Format_RGB888);
        this->channelmax=this->histo->getImage().spectrum()-1;
        this->clamp=this->histo->getClamp();
        this->currentpos=(T)0;
        draw();
    }

    void writeToData(Histotype& d)
    {
        d.setClamp(this->clamp);
    }

    void fromGraph(const QPointF &pt,const bool clicked)
    {
        if(!this->histo) return;
        const T pos=this->histo->fromHistogram((unsigned int)sofa::helper::round( pt.x()));
        if(!clicked)	this->currentpos=pos;
        else
        {
            if(cimg_library::cimg::abs(clamp[1]-pos)<cimg_library::cimg::abs(pos-clamp[0])) this->clamp[1]=pos;
            else this->clamp[0]=pos;
            emit clampModified();
        }

        draw();
    }

    void fromOption(const unsigned int i)
    {
        if(!this->histo) return;
        this->channel=i;
        draw();
    }

    void draw()
    {
        if(!this->histo) return;

        cimg.fill(255);
        cimg_forXY(this->histo->getImage(),x,y) if(this->histo->getImage()(x,y,0,this->channel)) this->cimg(0,x,y)=this->cimg(1,x,y)=this->cimg(2,x,y)=0;

        const unsigned int cmin=this->histo->toHistogram(this->clamp[0]),cpos=this->histo->toHistogram(this->currentpos),cmax=this->histo->toHistogram(this->clamp[1]);
        unsigned int nb = 0; if(cpos<(unsigned int)this->cimg.height()) nb = (int)this->histo->getHistogram()(cpos,0,0,this->channel);
        this->label1->setNum((double)this->clamp[0]);
        this->label2->setText(QString().setNum((double)this->currentpos) + " ( " + QString().setNum(nb) + " ) " );
        this->label3->setNum((double)this->clamp[1]);

        const unsigned char col[]= {0,0,100};
        const float opacity=0.5;
        for(unsigned int i=0; i<3; i++) this->cimg.draw_rectangle(i,0,0,i,cmin,this->cimg.depth(),col+i,opacity)
            .draw_rectangle(i,cmax,0,i,this->cimg.height(),this->cimg.depth(),col+i,opacity);

        const unsigned char currentcol[]= {200,100,0};
        for(unsigned int i=0; i<3; i++) this->cimg.draw_line(i,cpos,0,i,cpos,this->cimg.depth(),currentcol+i);

    }

    const unsigned int& getChannelMax() const {return channelmax;}

protected:
    const Histotype* histo;
    cimg_library::CImg<unsigned char> cimg; // Internal cimage memory shared with Qimage

    unsigned int channel;
    unsigned int channelmax;
    T currentpos;
    defaulttype::Vec<2,T> clamp;

    QLabel *label1;
    QLabel *label2;
    QLabel *label3;

};


//-----------------------------------------------------------------------------------------------//
//	image widget -> draw histogram and handle mouse events (clamping values)
//-----------------------------------------------------------------------------------------------//

class HistogramGraphScene : public QGraphicsScene
{
    Q_OBJECT

public:
    HistogramGraphScene(QImage* im,QObject *parent=0) : QGraphicsScene(parent)	,image(im)	{ this->setSceneRect(0,0,image->width(),image->height()); }

private:
    QImage *image;
    void drawBackground(QPainter *painter, const QRectF &rect)
    {
        QGraphicsScene::drawBackground(painter,rect);
        if(image) painter->drawImage(this->sceneRect(),*image);
    }
};


class HistogramGraphWidget : public QGraphicsView
{
    Q_OBJECT

public slots:
    void Render ()  { this->scene->update(); }
    void FitInView ()  { this->fitInView(this->sceneRect().x(),this->sceneRect().y(),this->sceneRect().width(),this->sceneRect().height());  }

public:
    HistogramGraphWidget(HistogramSetting *s, QWidget *parent)
        : QGraphicsView(parent) , S(s)
    {
        scene = new HistogramGraphScene(S->getImage(),this);
        this->setScene(scene);
        this->setMouseTracking(true);
        Render ();
    }

protected:
    void resizeEvent ( QResizeEvent * /*event*/ )  { FitInView(); }

    void mousePressEvent(QMouseEvent *mouseEvent)
    {
        QGraphicsView::mousePressEvent(mouseEvent);
        S->fromGraph(this->mapToScene ( mouseEvent->pos() ),true );
        Render ();
    }

    void mouseMoveEvent(QMouseEvent *mouseEvent)
    {
        QGraphicsView::mouseMoveEvent(mouseEvent);
        S->fromGraph(this->mapToScene ( mouseEvent->pos() ),false );
        Render ();
    }

    HistogramGraphScene * scene;
    HistogramSetting *S;
};


//-----------------------------------------------------------------------------------------------//
//	slider widget -> select channel (eg. rgb)
//-----------------------------------------------------------------------------------------------//

class HistogramOptionWidget: public QWidget
{
    Q_OBJECT

public:

    HistogramOptionWidget(HistogramSetting *s,HistogramGraphWidget *g, QWidget *parent)
        : QWidget(parent) , S(s) , G(g)
    {
        labelName=new QLabel(QString("Channel : "),this);
        slider=new QSlider(Qt::Horizontal,this);
        slider->setPageStep ( 1 );
        connect(slider, SIGNAL( valueChanged(int) ), this, SLOT( change(int) ) );

        label=new QLabel(this);
        label->setNum(0);

        QHBoxLayout *layout = new QHBoxLayout(this);
        layout->setMargin(0);
        layout->setSpacing(10);
        layout->addWidget(labelName);
        layout->addWidget(slider);
        layout->addWidget(label);
    }

    void setRange(const int minimum, const int maximum) 	{	slider->setRange(minimum,maximum); }

public slots:

    void change(const int i)
    {
        S->fromOption(i);
        G->Render();
        label->setNum(i);
    }

protected:
    QLabel *labelName;
    QSlider *slider;
    QLabel *label;
    HistogramSetting *S;
    HistogramGraphWidget *G;
};



//-----------------------------------------------------------------------------------------------//
//	Widget Container
//-----------------------------------------------------------------------------------------------//

template<class T>
class histogram_data_widget_container
{
public:
    typedef T Histotype;
    typedef THistogramSetting<Histotype> Setting;
    typedef HistogramOptionWidget Options;
    typedef HistogramGraphWidget Graph;
    typedef QVBoxLayout Layout;

    Setting* setting;
    Graph* graph;
    Options *options;
    Layout* container_layout;

    histogram_data_widget_container() : setting(NULL), graph(NULL), options(NULL), container_layout(NULL) {}

    bool createLayout( DataWidget* parent )
    {
        if( parent->layout() != NULL || container_layout != NULL ) return false;
        container_layout = new Layout(parent);
        return true;
    }

    bool createLayout( QLayout* layout)
    {
        if ( container_layout != NULL ) return false;
        container_layout = new Layout();
        layout->addItem(container_layout);
        return true;
    }

    bool createWidgets(DataWidget* parent, const Histotype& d, bool /*readOnly*/)
    {
        setting = new Setting(parent);
        setting->readFromData(d);

        graph = new Graph(setting,parent);

        if(setting->getChannelMax())
        {
            options = new Options(setting,graph,parent);
            options->setRange(0,setting->getChannelMax());
        }
        else options=NULL;

        return true;
    }

    void setReadOnly(bool /*readOnly*/) { }
    void readFromData(const Histotype& d0) {   setting->readFromData(d0); }
    void writeToData(Histotype& d) { setting->writeToData(d); }

    void insertWidgets()
    {
        assert(container_layout);
        if(graph) container_layout->addWidget(graph);
        if(setting) container_layout->addWidget(setting->getWidget());
        if(options) container_layout->addWidget(options);
    }
};


template<class T>
class SOFA_IMAGE_GUI_API HistogramDataWidget : public SimpleDataWidget<T, histogram_data_widget_container< T > >
{
public:
    typedef SimpleDataWidget<T, histogram_data_widget_container< T > > Inherit;
    typedef sofa::core::objectmodel::Data<T> MyData;
public:
    HistogramDataWidget(QWidget* parent,const char* name, MyData* d) : Inherit(parent,name,d) {}
    virtual unsigned int sizeWidget() {return 8;}
    virtual unsigned int numColumnWidget() {return 1;}
    virtual bool createWidgets()
    {
        bool b = Inherit::createWidgets();
        HistogramSetting* s = dynamic_cast<HistogramSetting*>(this->container.setting);
        this->connect(s,SIGNAL(clampModified()), this, SLOT(setWidgetDirty()));
        return b;
    }
};

}

}

}

#endif // SOFA_IMAGE_HISTOGRAMWIDGET_H
