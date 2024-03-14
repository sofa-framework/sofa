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
#include <QtCharts/QValueAxis>

#include <iostream>
#include <sofa/helper/AdvancedTimer.h>
#include <deque>
#include <unordered_map>


#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
using namespace QtCharts;
#endif

namespace sofa::gui::qt
{

typedef sofa::helper::system::thread::ctime_t ctime_t;

/**
 * @brief The ProfilerChartView class is a overide of QtCharts::QChartView
 * to be able to catch mouse selection and update all widgets of \sa SofaWindowProfiler
 * Will also overide drawForeground to draw a line to show the selected step.
 */
class ProfilerChartView : public QChartView
{
    Q_OBJECT
public:
    ProfilerChartView(QChart *chart, QWidget *parent, int bufferSize);

    /// method to update the max value of the Y axis (for line rendering).
    void updateYMax(int y) {m_maxY = y;}

protected:
    /// Overide to catch mouse selection on the graph.
    virtual void mousePressEvent(QMouseEvent *event);
    /// Overide to draw line at the step selected.
    virtual void drawForeground(QPainter *painter, const QRectF &rect);

signals:
    /// signal emited when a step has been selected on the graph @param int is the step number
    void pointSelected(int);

public slots:
    /// method to update the selection on the graph.
    void updateSelection(int x);

protected:
    /// copy of the serie size to check if selection is not out of bound
    int m_bufferSize;

    /// 2D point of the line to draw the selection
    QPointF m_lineSelect;
    QPointF m_lineOrigin;

    /// Step number selected on the graph. -1 if none
    int m_pointSelected;
    /// Stored value of the Y axis max.
    int m_maxY;
};

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

    /// method called when window is shown to activate AdvancedTimer recording.
    void activateATimer(bool activate);

    /// main method to iterate on the AdvancedTimer Data and update the info in the widgets
    void pushStepData();

    /// Method to clear all Data and reset graph
    void resetGraph();

    /**
     * @brief The AnimationSubStepData Internal class to store data for each step of the animation. Correspond to one AdvancedTimer::begin/end
     * Data stored/computed will be step name, its time in ms and the corresponding % inside the whole step.
     * the total ms and percentage it represent if this step has substeps.
     * Buffer of AnimationSubStepData corresponding to its children substeps
     */
    class AnimationSubStepData
    {
    public:
        AnimationSubStepData(int level, std::string name, ctime_t start);
        virtual ~AnimationSubStepData();

        int m_level;        
        std::string m_name;
        int m_nbrCall;
        ctime_t m_start;
        ctime_t m_end;

        std::string m_tag;
        SReal m_totalMs;
        SReal m_totalPercent;
        SReal m_selfMs;
        SReal m_selfPercent;

        void computeTimeAndPercentage(SReal invTotalMs);
        // Method to get a given step duration (ms) given its name and parent name
        SReal getStepMs(const std::string& stepName, const std::string& parentName);

        sofa::type::vector<AnimationSubStepData*> m_children;
    };

    /**
     * @brief The AnimationStepData internal class to store all info of a animation step recorded by AdvancedTimer
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
            , m_overheadMs(0.)
        {}

        AnimationStepData(int step, const std::string& idString);

        // Method to get a given step duration (ms) given its name and parent name
        SReal getStepMs(const std::string& stepName, const std::string& parentName);

        virtual ~AnimationStepData();
        int m_stepIteration;
        SReal m_totalMs;
        SReal m_selfMs {}; ///< Difference between the total time and the time of all children
        SReal m_selfPercent {}; ///< Difference between the total time and the time of all children as a percentage
        std::string m_idString; ///< Name of the timer

        sofa::type::vector<AnimationSubStepData*> m_subSteps;

        /// The overhead due to timers processing. In milliseconds
        SReal m_overheadMs;

        /// Total number of timers in this step
        unsigned int m_totalTimers {};
    protected:
        bool processData(const std::string& idString);
    };

protected:
    /// Method called at creation to init the chart
    void createChart();
    /// Method called at creation to init the QTreeWidget
    void createTreeView();

    /// Method called at each iteration to update the chart
    void updateChart();
    /// Method to add new QTreeWidgetItem item inside the QTreeWidget using the data from \sa AnimationSubStepData
    QTreeWidgetItem* addTreeItem(AnimationSubStepData* subStep);

    QTreeWidgetItem* addTreeItem(const AnimationStepData* step);

public slots:
    void closeEvent( QCloseEvent* ) override
    {
        emit(closeWindow(false));
    }

    /// Method to update all widgets from select absisse on the graph
    void updateFromSelectedStep(int step);

    /// Method called when a given @param step is triggered to update summary information
    void updateSummaryLabels(int step);
    /// Method called when a given @param step is triggered to update the QTreeView
    void updateTree(int step);
    /// Method called when a QTreeWidgetItem is selected in the Tree view.
    void onStepSelected(QTreeWidgetItem *item, int column);

    void expandRootNodeOnly() const;

signals:
    void closeWindow(bool);

protected:
    /// Pointer to the chart Data
    QChart *m_chart;
    /// Pointer to Y Axis
    QValueAxis *m_axisY;
    /// Pointer to the \sa ProfilerChartView class to handle chart drawing/selection
    ProfilerChartView* m_chartView;

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
    QLineSeries *m_series;

    /// Serie of selection substep duration in ms to be plot on the graph. size = \sa m_bufferSize
    QLineSeries *m_selectionSeries;

    struct CheckedSeries
    {
        QLineSeries* lineSeries;
        std::string checkedParentStep;
    };
    std::unordered_map<std::string, CheckedSeries> m_checkedSeries;

    /// Name of the substep selected in the Tree
    std::string m_selectedStep;
    /// Name of the parent of the substep selected in the Tree
    std::string m_selectedParentStep;
};

} //namespace sofa::gui::qt
