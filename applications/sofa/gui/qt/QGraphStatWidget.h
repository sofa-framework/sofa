/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_QGRAPHSTATWIDGET_H
#define SOFA_GUI_QT_QGRAPHSTATWIDGET_H

#include <sofa/simulation/common/Node.h>

#ifdef SOFA_QT4
#include <QWidget>
#include <QTextEdit>
#include <Q3GroupBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#else
#include <qwidget.h>
#include <qtextedit.h>
#include <qgroupbox.h>
#include <qlayout.h>
#include <qlabel.h>
#endif

#include <qwt_legend.h>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>

#ifndef SOFA_QT4
typedef QGroupBox Q3GroupBox;
typedef QTextEdit Q3TextEdit;
#endif

namespace sofa
{
namespace gui
{
namespace qt
{

/// Base class to make graphes in the stat tab of the node widget
/// TODO add gnuplot export
class QGraphStatWidget : public QWidget
{

    Q_OBJECT

public:

    QGraphStatWidget( QWidget* parent, simulation::Node* node, const QString& title, unsigned numberOfCurves )
        : QWidget( parent )
        , _numberOfCurves( numberOfCurves )
        , _node( node )
    {
        QVBoxLayout *layout = new QVBoxLayout( this, 0, 1, QString( "tabStats" ) + title );

#ifdef SOFA_QT4
        _graph = new QwtPlot( QwtText( title ), this );
#else
        _graph = new QwtPlot( this, title );
#endif

        _graph->setAxisTitle( QwtPlot::xBottom, "Time (s)" );
        _graph->setTitle( title );
        _graph->insertLegend( new QwtLegend(), QwtPlot::BottomLegend );

        layout->addWidget( _graph );

        _curves.resize( _numberOfCurves );
        _YHistory.resize( _numberOfCurves );
    }

    /// the only function that should be overloaded
    virtual void step()
    {
        //Add Time
        _XHistory.push_back( _node->getTime() );
    }

    void updateVisualization()
    {
        for( unsigned i=0 ; i<_numberOfCurves ; ++i )
            _curves[i]->setRawSamples( &_XHistory[0], &(_YHistory[i][0]), _XHistory.size() );
        _graph->replot();
    }


protected:

    /// set the index-th curve (index must be < _numberOfCurves)
    void setCurve( unsigned index, const QString& name, const QColor& color )
    {
        assert( index<_numberOfCurves );
        _curves[index] = new QwtPlotCurve( name );
        _curves[index]->attach( _graph );
        _curves[index]->setPen( QPen( color ) );
    }

    unsigned _numberOfCurves;

    simulation::Node *_node;

    std::vector< double > _XHistory; ///< X-axis values (by default take the node time)
    std::vector< std::vector< double > > _YHistory; ///< Y-axis values, one for each curve (_numberOfCurves)


    QwtPlot *_graph;
    std::vector< QwtPlotCurve* > _curves; ///< resized to _numberOfCurves
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QGRAPHSTATWIDGET_H

