/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include "QGraphStatWidget.h"


namespace sofa
{
namespace gui
{
namespace qt
{

QGraphStatWidget::QGraphStatWidget( QWidget* parent, simulation::Node* node, const QString& title, unsigned numberOfCurves )
    : QWidget( parent )
    , _numberOfCurves( numberOfCurves )
    , _node( node )
{
//        QVBoxLayout *layout = new QVBoxLayout( this, 0, 1, QString( "tabStats" ) + title );
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setMargin(0);
    layout->setSpacing(1);
    layout->setObjectName(QString( "tabStats" ) + title);

    _graph = new QwtPlot( QwtText( title ), this );


    _graph->setAxisTitle( QwtPlot::xBottom, "Time (s)" );
    _graph->setTitle( title );
    _graph->insertLegend( new QwtLegend(), QwtPlot::BottomLegend );

    layout->addWidget( _graph );

    _curves.resize( _numberOfCurves );
    _YHistory.resize( _numberOfCurves );
}

QGraphStatWidget::~QGraphStatWidget()
{
    delete _graph;
}

void QGraphStatWidget::step()
{
    //Add Time
    _XHistory.push_back( _node->getTime() );
}

void QGraphStatWidget::updateVisualization()
{
    for( unsigned i=0 ; i<_numberOfCurves ; ++i )
        _curves[i]->setRawSamples( &_XHistory[0], &(_YHistory[i][0]), _XHistory.size() );
    _graph->replot();
}


void QGraphStatWidget::setCurve( unsigned index, const QString& name, const QColor& color )
{
    assert( index<_numberOfCurves );
    _curves[index] = new QwtPlotCurve( name );
    _curves[index]->attach( _graph );
    _curves[index]->setPen( QPen( color ) );
}

} // qt
} // gui
} //sofa


