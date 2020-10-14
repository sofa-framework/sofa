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
#ifndef SOFA_GUI_QT_QGRAPHSTATWIDGET_H
#define SOFA_GUI_QT_QGRAPHSTATWIDGET_H

#include <sofa/simulation/Node.h>

#include <QWidget>
#include <QTextEdit>
#include <QGroupBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>


namespace QtCharts
{
    class QChartView;
    class QChart;
    class QLineSeries;
    class QValueAxis;
}

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
    QGraphStatWidget( QWidget* parent, simulation::Node* node, const QString& title, unsigned numberOfCurves, int bufferSize );
    virtual ~QGraphStatWidget();

    /// Main method called to update the graph
    virtual void step() final;

    /// the only function that should be overloaded
    virtual void stepImpl() = 0;

protected:
    /// set the index-th curve (index must be < _numberOfCurves)
    void setCurve( unsigned index, const QString& name, const QColor& color );

    /// Method to update Y axis scale
    void updateYAxisBounds(SReal value);

    /// flush data from series not anymore displayed
    void flushSeries();

    /// pointer to the node monitored
    simulation::Node *m_node;

    /// size of the buffers to stored
    int m_bufferSize;

    /// Pointer to the chart Data
    QtCharts::QChart *m_chart;

    /// vector of series to be ploted
    std::vector< QtCharts::QLineSeries *> m_curves;

    /// x axis pointer
    QtCharts::QValueAxis* m_axisX;
    /// y axis pointer
    QtCharts::QValueAxis* m_axisY;
    
    /// min y axis value stored
    SReal m_yMin;
    /// max y axis value stored
    SReal m_yMax;
    /// last timestep monitored
    SReal m_lastTime;
    /// step counter monitored
    int m_cptStep;
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QGRAPHSTATWIDGET_H

