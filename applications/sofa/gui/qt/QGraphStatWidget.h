/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <qwt_legend.h>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>


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

    QGraphStatWidget( QWidget* parent, simulation::Node* node, const QString& title, unsigned numberOfCurves );
    virtual ~QGraphStatWidget();
    /// the only function that should be overloaded
    virtual void step();

    void updateVisualization();


protected:

    /// set the index-th curve (index must be < _numberOfCurves)
    void setCurve( unsigned index, const QString& name, const QColor& color );
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

