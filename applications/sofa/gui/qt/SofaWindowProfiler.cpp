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
#include "SofaWindowProfiler.h"

#include <QHeaderView>
#include <QMenu>
#include <QMessageBox>

#include <QGridLayout>
#include <QDebug>
#include <QValueAxis>

namespace sofa
{

namespace gui
{

namespace qt
{
using namespace sofa::helper;
using namespace QtCharts;

///////////////////////////////////////// ProfilerChartView ///////////////////////////////////

ProfilerChartView::ProfilerChartView(QChart *chart, QWidget *parent, int bufferSize)
    : QChartView(chart, parent)
    , m_bufferSize(bufferSize)
    , m_pointSelected(-1)
    , m_maxY(1000)
{

}

void ProfilerChartView::mousePressEvent(QMouseEvent *event)
{
    auto const valueInSeries = chart()->mapToValue(event->localPos());

    int width = valueInSeries.x();

    if (width >= 0 && width < m_bufferSize)
    {
        m_pointSelected = width;
        updateSelection(m_pointSelected);
        emit pointSelected(m_pointSelected);
    }
    else
        m_pointSelected = -1;
}

void ProfilerChartView::updateSelection(int x)
{
    m_pointSelected = x;
    m_lineSelect = chart()->mapToPosition(QPointF(x, m_maxY));
    m_lineOrigin = chart()->mapToPosition(QPointF(x, 0));

    this->scene()->update(this->sceneRect());
}


void ProfilerChartView::drawForeground(QPainter *painter, const QRectF &)
{    
    if (m_pointSelected == -1)
        return;

    painter->drawLine(m_lineOrigin, m_lineSelect);
}





///////////////////////////////////////// AnimationSubStepData ///////////////////////////////////

SofaWindowProfiler::AnimationSubStepData::AnimationSubStepData(int level, std::string name, SReal selfMs)
    : m_level(level)
    , m_subStepName(name)
    , m_totalMs(selfMs)
    , m_selfMs(selfMs)
{

}

SofaWindowProfiler::AnimationSubStepData::~AnimationSubStepData()
{
    for (unsigned int i=0; i<m_children.size(); ++i)
        delete m_children[i];

    m_children.clear();
}

void SofaWindowProfiler::AnimationSubStepData::addChild(AnimationSubStepData* child)
{
    if (m_level + 1 == child->m_level) // direct child
    {
        m_children.push_back(child);
    }
    else //little child
    {
        if (m_children.empty()) // little child without child...
        {
            msg_error("SofaWindowProfiler") << "Problem when registering child: " << child->m_subStepName << " under parent: " << this->m_subStepName;

            // quick adoption
            AnimationSubStepData* achild = new AnimationSubStepData(m_level+1, "Step not registered", 0.0);
            m_children.push_back(achild);
        }

        m_children.back()->addChild(child);
    }
    // will update the selfMS and percentage after tree has been built.
}


void SofaWindowProfiler::AnimationSubStepData::computeTimeAndPercentage(SReal invTotalMs)
{
    if (!m_children.empty()) // compute from leaf to trunk
    {
        SReal totalChildrenMs = 0.0;
        for (unsigned int i=0; i<m_children.size(); i++)
        {
            m_children[i]->computeTimeAndPercentage(invTotalMs);
            totalChildrenMs += m_children[i]->m_totalMs;
        }

        // now that all children are update, compute ms and %
        m_selfMs = m_totalMs - totalChildrenMs;

        m_selfPercent = m_selfMs * invTotalMs;
        m_totalPercent = m_totalMs * invTotalMs;

//        std::cout << m_subStepName << " -> m_selfMs: " << m_selfMs << " - " << m_selfPercent
//                  << " | m_totalMs: " << m_totalMs << " - " << m_totalPercent << std::endl;
    }
    else // leaf
    {
        if (m_totalMs != m_selfMs)
            msg_warning("SofaWindowProfiler") << "m_totalMs: " << m_totalMs << " != m_selfMs: " << m_selfMs;

        // compute %
        m_selfPercent = m_selfMs * invTotalMs;
        m_totalPercent = m_totalMs * invTotalMs;

//        std::cout << m_subStepName << " -> m_selfMs: " << m_selfMs << " - " << m_selfPercent
//                  << " | m_totalMs: " << m_totalMs << " - " << m_totalPercent << std::endl;
    }
}

// quick method to convert freq time into ms
SReal convertInMs(ctime_t t, int nbIter=1)
{
    static SReal timer_freqd = SReal(CTime::getTicksPerSec());
    return 1000.0 * SReal(t) / SReal (timer_freqd * nbIter);
}



///////////////////////////////////////// AnimationStepData ///////////////////////////////////

SofaWindowProfiler::AnimationStepData::AnimationStepData(int step, helper::vector<AdvancedTimer::IdStep> _steps, std::map<AdvancedTimer::IdStep, sofa::helper::StepData> _stepData)
    : m_stepIteration(step)
    , m_totalMs(0.0)
{
    m_subSteps.clear();

    AnimationSubStepData* currentSubStep = nullptr;

    bool totalSet = false;
    for (unsigned int i=0; i<_steps.size(); i++)
    {        
        StepData& data = _stepData[_steps[i]];

        if (data.level == 0) // main info
        {
            m_totalMs = convertInMs(data.ttotal);
            if (m_totalMs != 0.0) // total Ms not always set.. need to understand why
                totalSet = true;
            //std::cout << " -> totalMs: " << m_totalMs << std::endl;
        }
        else if (data.level == 1) // direct substep
        {
            currentSubStep = new AnimationSubStepData(data.level, data.label, convertInMs(data.ttotal));
            m_subSteps.push_back(currentSubStep);

            if (!totalSet) // total Ms not always set.. sum totalMass of top level steps
                m_totalMs += currentSubStep->m_totalMs;
        }
        else
        {
            AnimationSubStepData* child = new AnimationSubStepData(data.level, data.label, convertInMs(data.ttotal));
            currentSubStep->addChild(child);
        }
    }

    // update percentage
    SReal invTotalMs = 100 / m_totalMs;
    for (unsigned int i=0; i<m_subSteps.size(); i++)
    {
        m_subSteps[i]->computeTimeAndPercentage(invTotalMs);
    }
}

SofaWindowProfiler::AnimationStepData::~AnimationStepData()
{
    for (unsigned int i=0; i<m_subSteps.size(); ++i)
    {
        delete m_subSteps[i];
        m_subSteps[i] = nullptr;
    }
    m_subSteps.clear();
}



///////////////////////////////////////// SofaWindowProfiler ///////////////////////////////////

SofaWindowProfiler::SofaWindowProfiler(QWidget *parent)
    : QDialog(parent)
    , m_step(0)
    , m_bufferSize(100)
    , m_maxFps(0)
    , m_fpsMaxAxis(0)
{
    setupUi(this);

    // fill buffer with empty data.
    m_profilingData.resize(m_bufferSize);
    for (unsigned int i=0; i<m_bufferSize; ++i)
        m_profilingData[i] = new AnimationStepData();

    // creating chart widget
    createChart();

    // create treeView
    createTreeView();

    // create and connect different widgets
    step_scroller->setRange(0, m_bufferSize-1);
    step_scroller->setMinimumWidth(200);
    step_scroller->setMaximumWidth(200);
    connect(step_scroller, SIGNAL(valueChanged(int)), this, SLOT(updateSummaryLabels(int)));
    connect(step_scroller, SIGNAL(valueChanged(int)), this, SLOT(updateTree(int)));

    connect(step_scroller, SIGNAL(valueChanged(int)), m_chartView, SLOT(updateSelection(int)));
    connect(m_chartView, SIGNAL(pointSelected(int)), this, SLOT(updateFromSelectedStep(int)));
}


void SofaWindowProfiler::pushStepData()
{
    m_profilingData.pop_front();
    m_profilingData.push_back(new AnimationStepData(m_step, sofa::helper::AdvancedTimer::getSteps("Animate", true), sofa::helper::AdvancedTimer::getStepData("Animate")));
    m_step++;
    sofa::helper::AdvancedTimer::clearData("Animate");
    updateChart();
}


void SofaWindowProfiler::createTreeView()
{
    // set column names
    QStringList columnNames;
    columnNames << "Hierarchy Step Name" << "Total (%)" << "Self (%)" << "Time (ms)" << "Self (ms)";
    tree_steps->setHeaderLabels(columnNames);

    // set column properties
    tree_steps->header()->setStretchLastSection(false);
    tree_steps->header()->setSectionResizeMode(0, QHeaderView::Stretch);
}


void SofaWindowProfiler::createChart()
{
    m_series = new QLineSeries();  

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_series->append(i, 0.0f);
    }

    m_chart = new QChart();
    m_chart->legend()->hide();
    m_chart->addSeries(m_series);
    QValueAxis *axisY = new QValueAxis();
    m_chart->addAxis(axisY, Qt::AlignLeft);
    m_series->attachAxis(axisY);

    m_chart->setTitle("Animation step duration (in ms)");
    m_chart->axisY()->setRange(0, 1000);

    m_chartView = new ProfilerChartView(m_chart, this, m_bufferSize);
    m_chartView->setRenderHint(QPainter::Antialiasing);

    Layout_graph->addWidget(m_chartView);
}


void SofaWindowProfiler::updateChart()
{
    bool updateAxis = false;

    // Need to slide all the serie. Sure this could be optimised with deeper knowledge in QLineSeries/QChart
    int cpt = 0;
    for (auto* stepData : m_profilingData)
    {
        m_series->replace(cpt, cpt, stepData->m_totalMs);

        if (m_fpsMaxAxis < stepData->m_totalMs){
            m_fpsMaxAxis = stepData->m_totalMs;
            updateAxis = true;
        }

        // keep max ms value
        if (m_maxFps < stepData->m_totalMs)
            m_maxFps = stepData->m_totalMs;

        cpt++;
    }

    // if needed enlarge the Y axis to cover new data
    if (updateAxis){
        m_chart->axisY()->setRange(0, m_fpsMaxAxis*1.1);
        m_chartView->updateYMax(m_fpsMaxAxis*1.1);
    }

    // every loop on buffer size check if Y axis can be reduced
    if ((m_step% m_bufferSize) == 0)
    {
        if (m_maxFps < m_fpsMaxAxis)
            m_fpsMaxAxis = m_maxFps;

        m_maxFps = 0;
        m_chart->axisY()->setRange(0, m_fpsMaxAxis*1.1);
        m_chartView->updateYMax(m_fpsMaxAxis*1.1);
    }

 //   m_chartView->update();

    // update all widgets from value sliced
    updateSummaryLabels(step_scroller->value());
    updateTree(step_scroller->value());
}


void SofaWindowProfiler::updateFromSelectedStep(int step)
{
    // only update scroller as all the connection are already made from the scroller value change.
    step_scroller->setValue(step);
}


void SofaWindowProfiler::updateSummaryLabels(int step)
{
    const AnimationStepData* stepData = m_profilingData.at(step);
    label_stepValue->setText(QString::number(stepData->m_stepIteration));
    label_timeValue->setText(QString::number(stepData->m_totalMs));
}

void SofaWindowProfiler::updateTree(int step)
{
    const AnimationStepData* stepData = m_profilingData.at(step);

    //clear old values
    tree_steps->clear();

    // add new step items
    for (unsigned int i=0; i<stepData->m_subSteps.size(); i++)
    {
        addTreeItem(stepData->m_subSteps[i], nullptr);
    }
}

void SofaWindowProfiler::addTreeItem(AnimationSubStepData* subStep, QTreeWidgetItem* parent)
{
    // add item to the tree
    QTreeWidgetItem* treeItem = nullptr;
    if (parent == nullptr) // top item
        treeItem = new QTreeWidgetItem(tree_steps);
    else
        treeItem = new QTreeWidgetItem(parent);

    treeItem->setText(0, QString::fromStdString(subStep->m_subStepName));
    treeItem->setText(1, QString::number(subStep->m_totalPercent, 'g', 2));
    treeItem->setText(2, QString::number(subStep->m_selfPercent, 'g', 2));
    treeItem->setText(3, QString::number(subStep->m_totalMs));
    treeItem->setText(4, QString::number(subStep->m_selfMs));

    if (parent == nullptr)
    {
        treeItem->setExpanded(true);
        QFont font = QApplication::font();
        font.setBold(true);

        for (int i=0; i<treeItem->columnCount(); i++)
            treeItem->setFont(i, font);
    }

    // process children
    for (unsigned int i=0; i<subStep->m_children.size(); i++)
        addTreeItem(subStep->m_children[i], treeItem);
}


} // namespace qt

} // namespace gui

} // namespace sofa
