/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include "TableDataWidget.h"
#include "FileManagement.h"

#include <sofa/helper/system/SetDirectory.h>

#include <sofa/core/topology/TopologyData.h>

#include <QtCharts/QChartView>
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QPixmap>
#include <QFile>

#include <fstream>


#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
using namespace QtCharts;
#endif

namespace sofa::gui::qt
{

template<class T>
class QDataSeries : public QLineSeries
{
protected:
    const T* data0;
public:
    typedef vector_data_trait<T> trait;
    typedef typename trait::value_type value_type;
    typedef vector_data_trait<value_type> vtrait;
    typedef typename vtrait::value_type real_type;
    QDataSeries() : data0(NULL) {}
    QDataSeries(const QDataSeries<T>& c) 
        : QLineSeries()
        , data0(c.data0) 
    {
        parseData();
    }

    void setData(const T* p) 
    { 
        data0 = p; 
        parseData();
    }

    virtual QLineSeries* copy() const { return new QDataSeries<T>(*this); }
    virtual sofa::Size size() const
    {
        if (data0 == NULL)
            return 0;
        else
            return trait::size(*(data0));
    }

    virtual void parseData()
    {
        clear();
        m_xMin = m_yMin = 1000000000.0;
        m_xMax = m_yMax = -1000000000.0;

        for (sofa::Index i = 0; i < this->size(); i++)
        {
            QPointF data = sample(i);
            append(data);

            if (data.x() > m_xMax)
                m_xMax = data.x();
            if (data.x() < m_xMin)
                m_xMin = data.x();

            if (data.y() > m_yMax)
                m_yMax = data.y();
            if (data.y() < m_yMin)
                m_yMin = data.y();
        }
    }

    virtual QPointF sample (sofa::Index i) const
    {
        if (i >= size())
            return QPointF();
        else if (vtrait::size(*trait::get(*(data0), i)) < 2)
            return QPointF(i, (double)(*vtrait::get(*trait::get(*(data0), i), 0)));
        else
            return QPointF((double)(*vtrait::get(*trait::get(*(data0), i), 0)), (double)(*vtrait::get(*trait::get(*(data0), i), 1)));
    }    

    real_type m_xMin, m_yMin;
    real_type m_xMax, m_yMax;
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
    typedef QChartView Widget;
    typedef QDataSeries<curve_type> CurveData;

    GraphWidget(QWidget *parent)
    {
        m_chart = new QChart();
        w = new QChartView(m_chart, parent);
        w->setMinimumHeight(300);

        m_axisX = new QValueAxis();
        m_axisX->setRange(0, 100);

        m_axisY = new QValueAxis();
        m_chart->addAxis(m_axisX, Qt::AlignBottom);
        m_chart->addAxis(m_axisY, Qt::AlignLeft);
        m_chart->legend()->setAlignment(Qt::AlignBottom);
    }

    virtual ~GraphWidget() {};

    QWidget *getWidget() {return w;}

    void readFromData(const data_type& d0)
    {
        double minX = 1000000000.0;
        double maxX = -1000000000.0;
        double minY = 1000000000.0;
        double maxY = -1000000000.0;
        currentData=d0;
        const data_type& d = currentData;
        auto nbData = trait::size(d);

        for (int i = 0; i < nbData; ++i)
        {
            const curve_type* v = trait::get(d, i);
            const char* name = trait::header(d, i);
            QString sName;
            if (name && *name)
                sName = name;
            else
                sName = "Unknown_" + QString::number(m_curves.size());

            auto itM = m_curves.find(sName);
            CurveData* cdata;
            if (itM != m_curves.end())
            {
                cdata = itM->second;
            }
            else {
                // new curve data to register and plot
                cdata = new CurveData();
                m_curves[sName] = cdata;

                cdata->setName(sName);
                cdata->setPen(QColor::fromHsv(255 * i / nbData, 255, 255));
                m_chart->addSeries(cdata);

                cdata->attachAxis(m_axisY);
                cdata->attachAxis(m_axisX);
            }

            cdata->setData(v);
            // gather min and max over all curves
           if(cdata->m_xMin < minX) minX = cdata->m_xMin;
           if(cdata->m_xMax > maxX) maxX = cdata->m_xMax;
           if(cdata->m_yMin < minY) minY = cdata->m_yMin;
           if(cdata->m_yMax > maxY) maxY = cdata->m_yMax;
        }

        m_axisX->setRange(minX, maxX);
        m_axisY->setRange(minY, maxY);
    }

    void exportGNUPlot(const std::string &baseFileName) const
    {
        const int n = trait::size(currentData);
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
        const std::string filename= baseFileName + ".png";

        QPixmap p(w->size());
        QPainter *paint = new QPainter(&p);
        w->render(paint);

        QFile* file = new QFile(QString::fromStdString(filename));
        file->open(QIODevice::WriteOnly);
        p.save(file, "PNG");

        std::cerr << "Export Image: " << filename << std::endl;
    }

protected:
    std::string getCurveFilename(unsigned int idx) const
    {
        std::string name(trait::header(currentData,idx));
        std::replace(name.begin(),name.end(),' ', '_');
        return name;
    }

    QChartView* w;

    /// Pointer to the chart Data
    QChart *m_chart;

    /// x axis pointer
    QValueAxis* m_axisX;
    /// y axis pointer
    QValueAxis* m_axisY;

    /// vector of series to be ploted
    std::map<QString, CurveData*> m_curves;

    data_type currentData;
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
        const bool b = GraphDataWidget<T>::createWidgets();
        //typename GraphWidget<T>::Widget* w = dynamic_cast<typename GraphWidget<T>::Widget*>(this->container.w->getWidget());
        //if (w)
        //{
        //    w->setAxisScaleEngine(GraphWidget<T>::Widget::yLeft, new QwtLinearScaleEngine);
        //}
        return b;
    }
};

} //namespace sofa::gui::qt
