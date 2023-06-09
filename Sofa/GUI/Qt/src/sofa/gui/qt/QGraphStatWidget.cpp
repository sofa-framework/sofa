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

#include <sofa/gui/qt/QGraphStatWidget.h>
#include <sofa/simulation/Node.h>

namespace sofa::gui::qt
{

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
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(1);
    layout->setObjectName(QString( "tabStats" ) + title);

    m_chart = new QChart();
    QFont font;
    font.setPixelSize(18);
    m_chart->setTitleFont(font);
    m_chart->setTitle(title);

    m_axisX = new QValueAxis();       
    m_axisX->setRange(0, m_node->getDt()*m_bufferSize);
    m_axisX->setTitleText("Time (ms)");

    m_axisY = new QValueAxis();
    m_axisY->setTitleText("Value");
    
    m_chart->addAxis(m_axisX, Qt::AlignBottom);
    m_chart->addAxis(m_axisY, Qt::AlignLeft);
    m_chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView* chartView = new QChartView(m_chart, this);
    layout->addWidget(chartView);
    
    m_curves.resize(numberOfCurves);

}

QGraphStatWidget::~QGraphStatWidget()
{
    
}

void QGraphStatWidget::step()
{
    const SReal time = m_node->getTime();
    if (time <= m_lastTime)
        return;

    // call internal method to add Data into series
    stepImpl();

    if (m_cptStep > m_bufferSize) // start swipping the Xaxis
    {
        const qreal min = m_axisX->min() + m_node->getDt();
        m_axisX->setRange(min, time);        

        // flush series data not anymore display for memory storage
        if ((m_cptStep% m_bufferSize * 2) == 0)
        {
            flushSeries();
        }
    }

    if ((m_cptStep% m_bufferSize) == 0)
    {
        if (m_axisY->max() > m_yMax)
            m_axisY->setMax(m_yMax*1.01);

        if (m_axisY->min() < m_yMin)
            m_axisY->setMin(m_yMin*0.99);

        m_yMax = -10000;
        m_yMin = 10000;
    }

    m_lastTime = time;
    m_cptStep++;
}


void QGraphStatWidget::flushSeries()
{
    for (const auto serie : m_curves)
    {
        if (serie->count() >= m_bufferSize) {
            serie->removePoints(0, m_bufferSize);
        }
    }
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

    m_curves[index]->attachAxis(m_axisY);
    m_curves[index]->attachAxis(m_axisX);
}


void QGraphStatWidget::updateYAxisBounds(SReal value)
{
    if (value > m_axisY->max())
    {
        m_axisY->setMax(value*1.01);
    }
    if (value < m_axisY->min())
    {
        m_axisY->setMin(value*0.99);
    }

    // for record
    if (value > m_yMax)
    {
        m_yMax = value;
    }

    if (value < m_yMin)
    {
        m_yMin = value;
    }
}


} //namespace sofa::gui::qt
