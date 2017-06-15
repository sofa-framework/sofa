/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_GRAPHDATAWIDGET_H
#define SOFA_GUI_QT_GRAPHDATAWIDGET_H

#include "TableDataWidget.h"
#include "FileManagement.h"

#include <sofa/simulation/Simulation.h>
#include <sofa/helper/system/SetDirectory.h>

#include <SofaBaseTopology/TopologyData.h>
#include <qwt_compat.h>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_scale_engine.h>
#include <qwt_series_data.h>
// #include <qwt_plot_printfilter.h>
#include <qwt_plot_renderer.h>

#include <fstream>

namespace sofa
{

namespace gui
{

namespace qt
{

template<class T>
class QwtDataAccess : public QwtSeriesData< QPointF >
{
protected:
    const T* data0;
public:
    typedef vector_data_trait<T> trait;
    typedef typename trait::value_type value_type;
    typedef vector_data_trait<value_type> vtrait;
    typedef typename vtrait::value_type real_type;
    QwtDataAccess() : data0(NULL) {}
    QwtDataAccess(const QwtDataAccess<T>& c) : QwtSeriesData<QPointF>(c), data0(c.data0) {}

    void setData(const T* p) { data0 = p; }
    virtual QwtSeriesData<QPointF>* copy() const { return new QwtDataAccess<T>(*this); }
    virtual size_t size() const
    {
        if (data0 == NULL)
            return 0;
        else
            return trait::size(*(data0));
    }

    /*
    virtual double x (size_t i) const
    {
    if (i >= size())
        return 0.0;
    else if (vtrait::size(*trait::get(*(data0), i)) < 2)
        return (double)i;
    else
        return (double)(*vtrait::get(*trait::get(*(data0), i), 0));
    }
    virtual double y (size_t i) const
    {
    if (i >= size())
        return 0.0;
    else if (vtrait::size(*trait::get(*(data0), i)) < 2)
        return (double)(*vtrait::get(*trait::get(*(data0), i), 0));
    else
        return (double)(*vtrait::get(*trait::get(*(data0), i), 1));
    }
    */

    virtual QPointF sample (size_t i) const
    {
        if (i >= size())
            return QPointF();
        else if (vtrait::size(*trait::get(*(data0), i)) < 2)
            return QPointF(i, (double)(*vtrait::get(*trait::get(*(data0), i), 0)));
        else
            return QPointF((double)(*vtrait::get(*trait::get(*(data0), i), 0)), (double)(*vtrait::get(*trait::get(*(data0), i), 1)));
    }

    virtual QRectF boundingRect () const
    {
        if (size() == 0)
            return QRectF();

        real_type x, y , xMin, xMax, yMin, yMax;

        if (vtrait::size(*trait::get(*(data0), 0)) < 2)
        {
            x = xMin = xMax = 0;
            y = yMin = yMax = (*vtrait::get(*trait::get(*(data0), 0), 0));
        }
        else
        {
            x = xMin = xMax = (*vtrait::get(*trait::get(*(data0), 0), 0));
            y = yMin = yMax = (*vtrait::get(*trait::get(*(data0), 0), 1));
        }

        for (size_t i=1; i < size(); i++)
        {
            if (vtrait::size(*trait::get(*(data0), i)) < 2)
            {
                x = i;
                y = (*vtrait::get(*trait::get(*(data0), i), 0));
            }
            else
            {
                x = (*vtrait::get(*trait::get(*(data0), i), 0));
                y = (*vtrait::get(*trait::get(*(data0), i), 1));
            }

            if (x > xMax)
                xMax = x;
            else if (x < xMin)
                xMin = x;

            if (y > yMax)
                yMax = y;
            else if (y < yMin)
                yMin = y;
        }

        return QRectF(xMin, yMin, xMax-xMin, yMax-yMin);
    }
};

class GraphSetting
{
public:
    virtual ~GraphSetting() {};
    virtual void exportGNUPlot(const std::string &baseFileName) const=0;
    virtual void exportImage(const std::string &baseFileName) const=0;
};

class GraphOptionWidget: public QWidget
{
    Q_OBJECT
public:

    GraphOptionWidget(const std::string &dataName, GraphSetting *graphConf);
public slots:

    void openFindFileDialog();
    void exportGNUPlot();
    void exportImage();
//MOC_SKIP_END

    bool isCheckedBox() {
        return checkBox->isChecked();
    }

protected:
    unsigned idfile;
    QPushButton* exportGNUPLOTButton;
    QLineEdit *fileGNUPLOTLineEdit;
    QPushButton* findGNUPLOTFile;

    QCheckBox * checkBox;
    QPushButton* exportImageButton;
    QLineEdit *fileImageLineEdit;
    QPushButton* findImageFile;

    GraphSetting *graph;
};




template <class DataType>
class GraphWidget: public GraphSetting
{
public:
    typedef DataType data_type;
    typedef vector_data_trait<DataType> trait;
    typedef typename trait::value_type curve_type;
    typedef QwtPlot Widget;
    typedef QwtPlotCurve Curve;
    typedef QwtDataAccess<curve_type> CurveData;

    GraphWidget(QWidget *parent)
    {
        w = new Widget(QwtText(""), parent);

        w->insertLegend(new QwtLegend(), QwtPlot::BottomLegend);
        w->setAxisScaleEngine(Widget::yLeft, new QwtLogScaleEngine(10));
    }

    virtual ~GraphWidget() {};

    QWidget *getWidget() {return w;}

    /* QColor getColor(float h)
     {
    int i = int(h * 6) % 6;
    float f = h * 6 - floor(h * 6);
    int c1 = 255 - floor(255 * f);
    int c2 = floor(255 * f);

    switch(i) {
    case 0: return QColor(255, c2, 0);
    case 1: return QColor(c1, 255, 0);
    case 2: return QColor(0, 255, c2);
    case 3: return QColor(0, c1, 255);
    case 4: return QColor(c2, 0, 255);
    case 5: default: return QColor(255, 0, c1);
    }
     }*/

    void readFromData(const data_type& d0)
    {
        double minX = 1000000000.0;
        double maxX = -1000000000.0;
        double minY = 1000000000.0;
        double maxY = -1000000000.0;
        currentData=d0;
        const data_type& d = currentData;
        int s = curve.size();
        int n = trait::size(d);
        for (int i=0; i<n; ++i)
        {
            const curve_type* v = trait::get(d,i);
            const char* name = trait::header(d,i);
            Curve *c;
            CurveData* cd;

            if (i >= s)
            {
                QString s;
                if (name && *name) s = name;
                c = new Curve(s);
                c->attach(w);
                cd = new CurveData;
                curve.push_back(c);
                cdata.push_back(cd);
                s = i+1;
            }
            else
            {
                c = curve[i];
                cd = cdata[i];
                QString s;
                if (name && *name) s = name;
                if (s != c->title().text())
                    c->setTitle(s);
            }

            // c->setPen(getColor(i / (float)n));
            c->setPen(QColor::fromHsv(255*i/n, 255, 255));
            cd->setData(v);
            c->setData(cd);
            if(c->minXValue() < minX) minX = c->minXValue();
            if(c->maxXValue() > maxX) maxX = c->maxXValue();
            if(c->minYValue() < minY) minY = c->minYValue();
            if(c->maxYValue() > maxY) maxY = c->maxYValue();

            rect = rect.united(cdata[i]->boundingRect());
        }


        if (s != n)
        {
            for (int i=n; i < s; ++i)
            {
                Curve* c = curve[i];
                c->detach();
                delete c;	// Curve has ownership of the CurveData
            }
            curve.resize(n);
            cdata.resize(n);
            s = n;
        }
        if (n > 0 && minX <= maxX)
        {
            w->setAxisScale(Widget::yLeft, minY, maxY);
            w->setAxisScale(Widget::xTop, minX, maxX);
        }
        w->replot();
    }

    void exportGNUPlot(const std::string &baseFileName) const
    {
        int n = trait::size(currentData);
        for (int i=0; i<n; ++i)
        {
            const curve_type& v = *(trait::get(currentData,i));
            const std::string filename=baseFileName + std::string("_") + getCurveFilename(i) + std::string(".txt");
            std::cerr << "Export GNUPLOT file: " + filename << std::endl;
            std::ofstream gnuplotFile(filename.c_str());
            for (unsigned int idx=0; idx<v.size(); ++idx)
                gnuplotFile << idx << " " << v[idx] << "\n";
            gnuplotFile.close();
        }
    }

    void exportImage(const std::string &baseFileName) const
    {
        const std::string filename=baseFileName + ".svg";

        const float resolution = 72.f; // dpi
        const float inch2mm = 25.4f;

        QwtPlotRenderer renderer;
        renderer.setDiscardFlag(QwtPlotRenderer::DiscardNone);
        renderer.renderDocument(w, filename.c_str(), "svg", QSizeF(inch2mm * w->width() / resolution, inch2mm * w->height() / resolution), resolution);

        std::cerr << "Export Image: " << filename << std::endl;

        //	Qwt 5.2.0 Code
        //	QwtPlotPrintFilter filter;
        //	filter.setOptions(QwtPlotPrintFilter::PrintAll);
        //	QImage image(w->width(), w->height(),QImage::Format_RGB32);
        //	image.fill(0xffffffff); //white image
        //	w->print(image,filter);
        //	image.save(filename.c_str());
    }

protected:
    std::string getCurveFilename(unsigned int idx) const
    {
        std::string name(trait::header(currentData,idx));
        std::replace(name.begin(),name.end(),' ', '_');
        return name;
    }

    Widget* w;

    helper::vector<Curve*> curve;
    helper::vector<CurveData*> cdata;
    data_type currentData;
    QwtDoubleRect rect;

};



template<class T>
class graph_data_widget_container
{
public:
    typedef T data_type;
    typedef GraphWidget<T> Widget;
    typedef QVBoxLayout Layout;
    Widget* w;
    GraphOptionWidget *options;
    Layout* container_layout;


    graph_data_widget_container() : w(NULL), options(NULL), container_layout(NULL) {}


    bool createLayout( DataWidget* parent )
    {
        if( parent->layout() != NULL || container_layout != NULL )
        {
            return false;
        }
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

    bool createWidgets(DataWidget* parent, const data_type& d, bool /*readOnly*/)
    {
        w = new Widget(parent);
        options = new GraphOptionWidget(parent->getBaseData()->getName(),w);
        w->readFromData(d);
        return true;
    }
    void setReadOnly(bool /*readOnly*/)
    {
    }
    void readFromData(const data_type& d0)
    {
        w->readFromData(d0);
        if (options->isCheckedBox()) options->exportImage();
    }
    void writeToData(data_type& /*d*/)
    {
    }

    void insertWidgets()
    {
        assert(container_layout);
        container_layout->addWidget(w->getWidget());
        container_layout->addWidget(options);
    }
};

template<class T>
class GraphDataWidget : public SimpleDataWidget<T, graph_data_widget_container< T > >
{
public:
    typedef SimpleDataWidget<T, graph_data_widget_container< T > > Inherit;
    typedef sofa::core::objectmodel::Data<T> MyData;
public:
    GraphDataWidget(QWidget* parent,const char* name, MyData* d) : Inherit(parent,name,d) {}
    virtual unsigned int sizeWidget() {return 8;}
    virtual unsigned int numColumnWidget() {return 1;}
};

template<class T>
class GraphDataWidget_Linear : public GraphDataWidget< T >
{
public:
    typedef sofa::core::objectmodel::Data<T> MyData;
    GraphDataWidget_Linear(QWidget* parent,const char* name, MyData* d) : GraphDataWidget <T>(parent,name,d) { }
    virtual bool createWidgets()
    {
        bool b = GraphDataWidget<T>::createWidgets();
        typename GraphWidget<T>::Widget* w = dynamic_cast<typename GraphWidget<T>::Widget*>(this->container.w->getWidget());
        if(w) w->setAxisScaleEngine(GraphWidget<T>::Widget::yLeft, new QwtLinearScaleEngine);
        return b;
    }
};

} // namespace qt

} // namespace gui

} // namespace sofa


#endif
