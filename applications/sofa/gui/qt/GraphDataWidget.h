/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_GRAPHDATAWIDGET_H
#define SOFA_GUI_QT_GRAPHDATAWIDGET_H

#include <sofa/gui/qt/TableDataWidget.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/component/topology/PointData.h>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_scale_engine.h>

namespace sofa
{

namespace gui
{

namespace qt
{

template<class T>
class QwtDataAccess : public QwtData
{
protected:
    const T* data0;
public:
    typedef vector_data_trait<T> trait;
    typedef typename trait::value_type value_type;
    typedef vector_data_trait<value_type> vtrait;
    typedef typename vtrait::value_type real_type;
    QwtDataAccess() : data0(NULL) {}
    QwtDataAccess(const QwtDataAccess<T>& c) : QwtData(c), data0(c.data0) {}

    void setData(const T* p) { data0 = p; }
    virtual QwtData* copy() const { return new QwtDataAccess<T>(*this); }
    virtual size_t size() const
    {
        if (data0 == NULL)
            return 0;
        else
            return trait::size(*(data0));
    }
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
};

template<class T>
class graph_data_widget_container
{
public:
    typedef T data_type;
    typedef vector_data_trait<T> trait;
    typedef typename trait::value_type curve_type;
    typedef QwtPlot Widget;
    typedef QwtPlotCurve Curve;
    typedef QwtDataAccess<curve_type> CurveData;

    Widget* w;
    helper::vector<Curve*> curve;
    helper::vector<CurveData*> cdata;
    data_type currentData;
    QwtDoubleRect rect;
    graph_data_widget_container() : w(NULL) {}


    bool createWidgets(DataWidget* parent, const data_type& d, bool /*readOnly*/)
    {
        if (Q3GroupBox* box = dynamic_cast<Q3GroupBox*>(parent)) box->setColumns(1);
#ifdef SOFA_QT4
        w = new Widget(QwtText(""), parent);
#else
        w = new Widget(parent, "Graph");
#endif
        w->insertLegend(new QwtLegend(), QwtPlot::BottomLegend);
        w->setAxisScaleEngine(Widget::yLeft, new QwtLog10ScaleEngine);
        readFromData(d);
        return true;
    }
    void setReadOnly(bool /*readOnly*/)
    {
    }
    void readFromData(const data_type& d0)
    {
        currentData = d0;
        const data_type& d = currentData;
        int s = curve.size();
        int n = trait::size(d);
        for (int i=0; i<n; ++i)
        {
            const curve_type* v = trait::get(d,i);
            const char* name = trait::header(d,i);
            if (i >= s)
            {
                Curve *c;
                QString s;
                if (name && *name) s = name;
                c = new Curve(s);
                c->attach(w);
                switch(i % 6)
                {
                case 0 : c->setPen(QPen(Qt::red)); break;
                case 1 : c->setPen(QPen(Qt::green)); break;
                case 2 : c->setPen(QPen(Qt::blue)); break;
                case 3 : c->setPen(QPen(Qt::cyan)); break;
                case 4 : c->setPen(QPen(Qt::magenta)); break;
                case 5 : c->setPen(QPen(Qt::yellow)); break;
                }
                CurveData* cd = new CurveData;
                cd->setData(v);
                c->setData(*cd);
                curve.push_back(c);
                cdata.push_back(cd);
                s = i+1;
            }
            else
            {
                Curve* c = curve[i];
                CurveData* cd = cdata[i];
                QString s;
                if (name && *name) s = name;
                if (s != c->title().text())
                    c->setTitle(s);
                cd->setData(v);
                c->setData(*cd);
            }
            rect = rect.unite(cdata[i]->boundingRect());
        }
        if (s != n)
        {
            for (int i=n; i < s; ++i)
            {
                Curve* c = curve[i];
                CurveData* cd = cdata[i];
                c->detach();
                delete c;
                delete cd;
            }
            curve.resize(n);
            cdata.resize(n);
            s = n;
        }
        if (n > 0 && !rect.isNull())
        {
            w->setAxisScale(Widget::yLeft, rect.top(), rect.bottom());
            w->setAxisScale(Widget::xBottom, rect.left(), rect.right());
        }
        w->replot();
    }
    void writeToData(data_type& /*d*/)
    {
    }
    bool processChange(const QObject* sender)
    {
        if (sender == w)
            return true;
        return false;
    }
};

template<class T>
class GraphDataWidget : public SimpleDataWidget<T, graph_data_widget_container< T > >
{
public:
    typedef SimpleDataWidget<T, graph_data_widget_container< T > > Inherit;
    typedef sofa::core::objectmodel::TData<T> MyData;
public:
    GraphDataWidget(QWidget* parent,const char* name, MyData* d) : Inherit(parent,name,d) {}
    virtual unsigned int sizeWidget() {return 3;}
    virtual unsigned int numColumnWidget() {return 1;}
};

} // namespace qt

} // namespace gui

} // namespace sofa


#endif
