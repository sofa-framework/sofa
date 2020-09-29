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

#include "QGraphStatWidget.h"

namespace sofa
{
namespace gui
{
namespace qt
{

using namespace QtCharts;

QGraphStatWidget::QGraphStatWidget( QWidget* parent, simulation::Node* node, const QString& title, unsigned numberOfCurves, int bufferSize)
    : QWidget( parent )
    , m_node( node )
    , m_bufferSize(bufferSize)
    , m_yMin(10000)
    , m_yMax(-10000)
    , m_lastTime(0.0)
    , m_cptStep(0)    
{
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setMargin(0);
    layout->setSpacing(1);
    layout->setObjectName(QString( "tabStats" ) + title);

    m_chart = new QChart();
    m_chart->setTitle(title);

    m_axisX = new QValueAxis();       
    m_axisX->setRange(0, m_node->getDt()*m_bufferSize);
    m_axisX->setTitleText("Time (ms)");

    m_axisY = new QValueAxis();
    m_axisY->setTitleText("Value");
    
    m_chart->addAxis(m_axisX, Qt::AlignBottom);
    m_chart->addAxis(m_axisY, Qt::AlignLeft);
    m_chart->legend()->setAlignment(Qt::AlignBottom);

    m_chartView = new QChartView(m_chart, this);
    layout->addWidget(m_chartView);
    
    m_curves.resize(numberOfCurves);

}

QGraphStatWidget::~QGraphStatWidget()
{
    //delete _graph;
}

void QGraphStatWidget::step()
{
    SReal time = m_node->getTime();
    if (time <= m_lastTime)
        return;

    stepImpl();

    if (m_cptStep > m_bufferSize) // start swipping
    {
        qreal min = m_axisX->min() + m_node->getDt();
        m_axisX->setRange(min, time);

        if ((m_cptStep% m_bufferSize * 2) == 0)
        {
            reduceSeries();
        }
    }

    m_lastTime = time;
    m_cptStep++;
    
}

void QGraphStatWidget::reduceSeries()
{
    for (auto serie : m_curves)
    {
        serie->removePoints(0, m_bufferSize);
    }
}

void QGraphStatWidget::updateVisualization()
{
    //std::cout << "updateVisualization()" << std::endl;
   /* for( unsigned i=0 ; i<_numberOfCurves ; ++i )
        _curves[i]->setRawSamples( &_XHistory[0], &(_YHistory[i][0]), _XHistory.size() );
    _graph->replot();*/
}


void QGraphStatWidget::setCurve( unsigned index, const QString& name, const QColor& color )
{
    if (index >= m_curves.size())
    {
        m_curves.resize(index+1);
    }

    m_curves[index] = new QLineSeries();
    m_curves[index]->setName(name);
    m_curves[index]->setPen(QPen(color));
    
    m_chart->addSeries(m_curves[index]);    

    //for (unsigned int i = 0; i < m_bufferSize; i++)
    m_curves[index]->attachAxis(m_axisY);
    m_curves[index]->attachAxis(m_axisX);
   // m_chart->setAxisY(m_Xaxis, m_curves[index]);
}


} // qt
} // gui
} //sofa


