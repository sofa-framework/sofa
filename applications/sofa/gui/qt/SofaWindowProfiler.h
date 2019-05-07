/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_WINDOWPROFILER_H
#define SOFA_WINDOWPROFILER_H

#include <ui_WindowProfiler.h>
#include "PieWidget.h"
#include "QVisitorControlPanel.h"

#include <QTreeWidgetItem>
#include <QDrag>
#include <QPixmap>
#include <QTableWidget>
#include <QComboBox>

#include <QDialog>
#include <QPainter>
#include <QTableWidget>

#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>

#include <iostream>
#include <sofa/helper/AdvancedTimer.h>
#include <deque>

namespace sofa
{

namespace gui
{

namespace qt
{

/**
 * @brief The SofaWindowProfiler class
 * This class is a QDialog widget to display information recorded by AdvancedTimer mechanism
 * At each step, info will be gathered from the AdvancedTimer using class sofa::helper::StepData
 * Info will be displayed by:
 * - ploting the step duration into a graph
 * - Showing information duration/step number
 * - Showing all substep of an animation step with their own duration in ms and the corresponding percentage over the whole step.
 */
class SofaWindowProfiler: public QDialog, public Ui_WindowProfiler
{
    Q_OBJECT
public:
    SofaWindowProfiler(QWidget* parent);

    /// main method to iterate on the advanceTimer Data and update the info in the widgets
    void pushStepData();

    /**
     * @brief The AnimationSubStepData Internal class to store data for each step of the animation. Correspond to one AdvanceTimer::begin/end
     * Data stored/computed will be step name, its time in ms and the corresponding % inside the whole step.
     * the total ms and percentage it represent if this step has substeps.
     * Buffer of AnimationSubStepData corresponding to its children substeps
     */
    class AnimationSubStepData
    {
    public:
        AnimationSubStepData(int level, std::string name, SReal selfMs);
        virtual ~AnimationSubStepData();

        int m_level;
        std::string m_subStepName;
        SReal m_totalMs;
        SReal m_totalPercent;
        SReal m_selfMs;
        SReal m_selfPercent;

        void addChild(AnimationSubStepData* child);
        void computeTimeAndPercentage(SReal invTotalMs);

        sofa::helper::vector<AnimationSubStepData*> m_children;
    };

    /**
     * @brief The AnimationStepData internal class to store all info of a animation step recorded by advanceTimer
     * Data stored/computed will be the step number, and the total time in ms of the step.
     * All Data will then be stored inside a tree of \sa AnimationSubStepData tree.
     */
    class AnimationStepData
    {
    public:
        // default constructor for empty data.
        AnimationStepData()
            : m_stepIteration(-1)
            , m_totalMs(0.0)
        {}

        AnimationStepData(int step, helper::vector<helper::AdvancedTimer::IdStep> _steps, std::map<sofa::helper::AdvancedTimer::IdStep, sofa::helper::StepData> _stepData);

        virtual ~AnimationStepData();
        int m_stepIteration;
        SReal m_totalMs;

        sofa::helper::vector<AnimationSubStepData*> m_subSteps;
    };

protected:
    /// Method called at creation to init the chart
    void createChart();
    /// Method called at creation to init the QTreeWidget
    void createTreeView();

    /// Method called at each iteration to update the chart
    void updateChart();
    /// Method to add new QTreeWidgetItem item inside the QTreeWidget using the data from \sa AnimationSubStepData
    void addTreeItem(AnimationSubStepData* subStep, QTreeWidgetItem* parent);

public slots:
    void closeEvent( QCloseEvent* )
    {
        emit(closeWindow(false));
    }

    /// Method called when a given @param step is triggered to update summary information
    void updateSummaryLabels(int step);
    /// Method called when a given @param step is triggered to update the QTreeView
    void updateTree(int step);

signals:
    void closeWindow(bool);

protected:
    /// Pointer to the chart Data
    QtCharts::QChart *m_chart;
    /// Pointer to the chart drawing
    QtCharts::QChartView* m_chartView;

    /// Current animation step internally recorded.
    int m_step;
    /// Size of the buffer data stored. (i.e number of stepData info stored)
    unsigned int m_bufferSize;
    /// Bigger step encountered in ms.
    SReal m_maxFps;
    /// Current Y max value of the graph (max ms encountered x1.1)
    SReal m_fpsMaxAxis;

    /// Buffer of \sa AnimationStepData (data for each step), deque size correspond to \sa m_bufferSize
    std::deque<AnimationStepData*> m_profilingData;

    /// Serie of step duration in ms to be plot on the graph. size = \sa m_bufferSize
    QtCharts::QLineSeries *m_series;
};
}
}
}

#endif // SOFA_WINDOWPROFILER_H
