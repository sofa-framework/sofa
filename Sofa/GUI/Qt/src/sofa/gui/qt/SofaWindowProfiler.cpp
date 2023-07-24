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
#include "SofaWindowProfiler.h"

#include <stack>
#include <QHeaderView>
#include <QMenu>
#include <QMessageBox>
#include <sofa/helper/logging/Messaging.h>
#include <QGridLayout>
#include <QDebug>
#include <utility>

namespace sofa::gui::qt
{
using namespace sofa::helper;

typedef sofa::helper::system::thread::ctime_t ctime_t;
typedef sofa::helper::system::thread::CTime CTime;


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

    const int width = valueInSeries.x();

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



// quick method to convert freq time into ms
SReal convertInMs(ctime_t t, int nbIter=1)
{
    static SReal timer_freqd = SReal(CTime::getTicksPerSec());
    return 1000.0 * SReal(t) / SReal (timer_freqd * nbIter);
}

///////////////////////////////////////// AnimationSubStepData ///////////////////////////////////

SofaWindowProfiler::AnimationSubStepData::AnimationSubStepData(int level, std::string name, ctime_t start)
    : m_level(level)
    , m_name(std::move(name))
    , m_nbrCall(1)
    , m_start(start)
{

}

SofaWindowProfiler::AnimationSubStepData::~AnimationSubStepData()
{
    for (unsigned int i=0; i<m_children.size(); ++i)
        delete m_children[i];

    m_children.clear();
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
        m_totalMs = convertInMs(m_end - m_start);
        m_selfMs = m_totalMs - totalChildrenMs;

        m_selfPercent = m_selfMs * invTotalMs;
        m_totalPercent = m_totalMs * invTotalMs;
    }
    else // leaf
    {
        // compute ms:
        m_totalMs = convertInMs(m_end - m_start);
        if (m_nbrCall != 1)
            m_selfMs = SReal(m_totalMs / m_nbrCall);
        else
            m_selfMs = m_totalMs;

        // compute %
        m_selfPercent = m_selfMs * invTotalMs;
        m_totalPercent = m_totalMs * invTotalMs;
    }
}

SReal SofaWindowProfiler::AnimationSubStepData::getStepMs(const std::string& stepName, const std::string& parentName)
{
    SReal result = 0.0;
    if (parentName == m_name)
    {
        for (unsigned int i=0; i<m_children.size(); i++)
        {
            if (m_children[i]->m_name == stepName)
                return m_children[i]->m_totalMs;
        }
    }
    else
    {
        for (unsigned int i=0; i<m_children.size(); i++)
        {
            result = m_children[i]->getStepMs(stepName, parentName);
            if (result != 0.0)
                return result;
        }
    }

    return 0.0;
}



///////////////////////////////////////// AnimationStepData ///////////////////////////////////

SofaWindowProfiler::AnimationStepData::AnimationStepData(int step, const std::string& idString)
    : m_stepIteration(step)
    , m_totalMs(0.0)
    , m_idString(idString)
    , m_overheadMs(0.)
{
    m_subSteps.clear();

    const bool res = processData(idString);
    if (!res) // error clear data
    {
        for (unsigned int i=0; i<m_subSteps.size(); ++i)
        {
            delete m_subSteps[i];
            m_subSteps[i] = nullptr;
        }
        m_subSteps.clear();
    }
}


bool SofaWindowProfiler::AnimationStepData::processData(const std::string& idString)
{
    type::vector<Record> _records = sofa::helper::AdvancedTimer::getRecords(idString);

    m_totalTimers = 0;

    //AnimationSubStepData* currentSubStep = nullptr;
    std::stack<AnimationSubStepData*> processStack;
    int level = 0;
    ctime_t t0 = 0;
    const ctime_t tEnd = CTime::getTime();
    ctime_t tCurr;
    for (unsigned int ri = 0; ri < _records.size(); ++ri)
    {
        const Record& rec = _records[ri];

        if (level == 0) // main step
        {
            t0 = rec.time;
            level++;
            continue;
        }

        tCurr = rec.time - t0;

        if (rec.type == Record::RBEGIN || rec.type == Record::RSTEP_BEGIN || rec.type == Record::RSTEP)
        {
//            for (int i=0; i<level; ++i)
//                std::cout << ".";
//            std::cout << level << " Begin: " << rec.label << " at " << rec.time << " obj: " << rec.obj << " val: " << rec.val << std::endl;

            AnimationSubStepData* currentSubStep = new AnimationSubStepData(level, rec.label, tCurr);
            if (rec.obj)
                currentSubStep->m_tag = std::string(AdvancedTimer::IdObj(rec.obj));

            if (level == 1) // Add top level step
                m_subSteps.push_back(currentSubStep);
            else
            {
                if (processStack.empty())
                {
                    msg_error("SofaWindowProfiler") << "No parent found to add step: " << currentSubStep->m_name;
                    delete currentSubStep;
                    return false;
                }
                else if (processStack.top()->m_level + 1 != currentSubStep->m_level)
                {
                    msg_warning("SofaWindowProfiler") << "Problem of level coherence between step: " << currentSubStep->m_name << " with level: " << currentSubStep->m_level
                                                      << " and parent step: " << processStack.top()->m_name << " with level: " << processStack.top()->m_level;
                }

                // add next step to the hierarchy
                processStack.top()->m_children.push_back(currentSubStep);
            }

            // add step into the stack for parent/child order
            processStack.push(currentSubStep);
            ++level;
        }

        if (rec.type == Record::REND || rec.type == Record::RSTEP_END)
        {
            ++m_totalTimers;
            --level;
//            for (int i=0; i<level; ++i)
//                std::cout << ".";
//            std::cout << level << " End: " << rec.label << " at " << rec.time << " obj: " << rec.obj << " val: " << rec.val << std::endl;

            if (processStack.empty())
            {
                msg_error("SofaWindowProfiler") << "End step with no step in the stack for: " << rec.label;
                return false;
            }
            else if (rec.label != processStack.top()->m_name)
            {
                msg_error("SofaWindowProfiler") << "Not the same name to end step between logs: " << rec.label << " and top stack: " << processStack.top()->m_name;
                return false;
            }

            AnimationSubStepData* currentSubStep = processStack.top();
            processStack.pop();

            currentSubStep->m_end = tCurr;
        }
    }
    // compute total MS step:
    m_totalMs = convertInMs(tEnd - t0);

    // update percentage
    const SReal invTotalMs = 100. / m_totalMs;
    SReal totalChildrenMs = 0.0;
    for (unsigned int i=0; i<m_subSteps.size(); i++)
    {
        m_subSteps[i]->computeTimeAndPercentage(invTotalMs);
        totalChildrenMs += m_subSteps[i]->m_totalMs;
    }

    m_selfMs = m_totalMs - totalChildrenMs;
    m_selfPercent = 100. * m_selfMs / m_totalMs;

    return true;
}


SReal SofaWindowProfiler::AnimationStepData::getStepMs(const std::string& stepName, const std::string& parentName)
{
    SReal result = 0.0;
    if (parentName.empty())
    {
        for (unsigned int i=0; i<m_subSteps.size(); i++)
        {
            if (m_subSteps[i]->m_name == stepName)
                return m_subSteps[i]->m_totalMs;
        }
    }
    else
    {
        for (unsigned int i=0; i<m_subSteps.size(); i++)
        {
            result = m_subSteps[i]->getStepMs(stepName, parentName);
            if (result != 0.0)
                return result;
        }
    }

    return 0.0;
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
    : QDialog(parent, Qt::WindowFlags() | Qt::WindowMaximizeButtonHint | Qt::WindowCloseButtonHint)
    , m_step(0)
    , m_bufferSize(100)
    , m_maxFps(0)
    , m_fpsMaxAxis(0)
    , m_selectedStep("")
    , m_selectedParentStep("")
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

    ExpandAllButton->setIcon(QIcon(":/RealGUI/expandAll"));
    CollapseAllButton->setIcon(QIcon(":/RealGUI/collapseAll"));
    for (auto* button : {ExpandAllButton, CollapseAllButton})
    {
        button->setFixedWidth(button->height());
    }

    connect ( ExpandAllButton, SIGNAL ( clicked() ), tree_steps, SLOT ( expandAll() ) );
    connect ( CollapseAllButton, SIGNAL ( clicked() ), this, SLOT ( expandRootNodeOnly() ) );
}


void SofaWindowProfiler::activateATimer(bool activate)
{
    sofa::helper::AdvancedTimer::setEnabled("Animate", activate);
    sofa::helper::AdvancedTimer::setInterval("Animate", 1);
    sofa::helper::AdvancedTimer::setOutputType("Animate", "gui");
}


void SofaWindowProfiler::pushStepData()
{
    const ctime_t start = CTime::getRefTime();

    m_profilingData.pop_front();
    const static std::string idString = "Animate";
    m_profilingData.push_back(new AnimationStepData(m_step, idString));
    m_step++;

    updateChart();

    m_profilingData.back()->m_overheadMs = convertInMs(CTime::getRefTime() - start);
}


void SofaWindowProfiler::resetGraph()
{
    if (m_step == 0)
        return;

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_series->replace(i, 0.0, 0.0);
        m_selectionSeries->replace(i, 0.0, 0.0);
        m_selectionSeries->setName("Selected SubStep");
        if (m_profilingData[i] && m_profilingData[i]->m_stepIteration != -1)
        {
            delete m_profilingData[i];
            m_profilingData[i] = new AnimationStepData();
        }
    }

    step_scroller->setValue(1); // for rest by changing 2 times value
    step_scroller->setValue(0);
    m_step = 0;
    m_selectedStep = "";
    m_selectedParentStep = "";
}


void SofaWindowProfiler::createTreeView()
{
    //list of the columns description
    //- first: column names
    //- second: tooltip (description of the column)
    const std::vector< std::pair< QString, QString > > columnsLabels = {
            {"Hierarchy Step Name", "Label of the measured step"},
            {"Total (%)", "Percentage of duration of this step compared to the duration of the root step"},
            {"Self (%)", "- If the step has child steps: percentage of the duration "
                           "of this step minus the sum of durations of its children, compared to "
                           "the duration of the root step.\n"
                           "- If the step has no child step: percentage of the average duration "
                           "of this step in case of multiple calls of this step during this time step, "
                           "compared to the duration of the root step."},
            {"Time (ms)", "Duration in milliseconds of this step"},
            {"Self (ms)", "- If the step has child steps: duration in milliseconds of "
                             "this step minus the sum of durations of its children.\n"
                             "- If the step has no child step: average duration in milliseconds of "
                             "this step in case of multiple calls of this step during this time step."}
    };

    // set column names
    QStringList columnNames;

    for (std::size_t i = 0; i < columnsLabels.size(); ++i)
    {
        columnNames << columnsLabels[i].first;
        tree_steps->headerItem()->setToolTip(i, columnsLabels[i].second);
    }
    tree_steps->setHeaderLabels(columnNames);

    tree_steps->headerItem()->setToolTip(1, QString("Percentage of duration of this step compared to the duration of the root step"));
    tree_steps->headerItem()->setToolTip(2,
                                         QString("- If the step has child steps: percentage of the duration "
                                                 "of this step minus the sum of durations of its children, compared to "
                                                 "the duration of the root step.\n"
                                                 "- If the step has no child step: percentage of the average duration "
                                                 "of this step in case of multiple calls of this step during this time step, "
                                                 "compared to the duration of the root step."));
    tree_steps->headerItem()->setToolTip(3, QString("Duration in milliseconds of this step"));
    tree_steps->headerItem()->setToolTip(4,
                                         QString("- If the step has child steps: duration in milliseconds of "
                                                 "this step minus the sum of durations of its children.\n"
                                                 "- If the step has no child step: average duration in milliseconds of "
                                                 "this step in case of multiple calls of this step during this time step."));

    // set column properties
    tree_steps->header()->setStretchLastSection(false);
    tree_steps->header()->setSectionResizeMode(0, QHeaderView::Stretch);

    connect(tree_steps, SIGNAL(itemClicked(QTreeWidgetItem*,int)), this, SLOT(onStepSelected(QTreeWidgetItem*,int)));
}


void SofaWindowProfiler::createChart()
{
    m_series = new QLineSeries();
    QPen pen = m_series->pen();
    pen.setColor(Qt::red);
    pen.setWidth(2);
    m_series->setName("Full Animation Step");
    m_series->setPen(pen);

    m_selectionSeries = new QLineSeries();
    m_selectionSeries->setName("Selected SubStep");

    for(unsigned int i=0; i<m_bufferSize; i++)
    {
        m_series->append(i, 0.0f);
        m_selectionSeries->append(i, 0.0f);
    }

    m_chart = new QChart();
    m_chart->legend()->setShowToolTips(true);
    m_chart->addSeries(m_series);
    m_chart->addSeries(m_selectionSeries);
    m_axisY = new QValueAxis();
    m_chart->addAxis(m_axisY, Qt::AlignLeft);
    m_series->attachAxis(m_axisY);
    m_selectionSeries->attachAxis(m_axisY);

    m_chart->setTitle("Steps durations (in ms)");
    m_axisY->setRange(0, 1000);

    m_chartView = new ProfilerChartView(m_chart, this, m_bufferSize);
    m_chartView->setRenderHint(QPainter::Antialiasing);

    Layout_graph->addWidget(m_chartView);
}


void SofaWindowProfiler::updateChart()
{
    bool updateAxis = false;

    QVector<QPointF> seriesPoints;
    QVector<QPointF> selectedStepPoints;
    std::unordered_map<std::string, QVector<QPointF> > checkedSeriesPoints;

    int cpt = 0;
    for (auto* stepData : m_profilingData)
    {
        seriesPoints.push_back(QPointF(cpt, stepData->m_totalMs));

        if (!m_selectedStep.empty())
        {
            const SReal value = stepData->getStepMs(m_selectedStep, m_selectedParentStep);
            selectedStepPoints.push_back(QPointF(cpt, value));
        }

        for (const auto& checkedSeries : m_checkedSeries)
        {
            const SReal value = stepData->getStepMs(checkedSeries.first, checkedSeries.second.checkedParentStep);
            checkedSeriesPoints[checkedSeries.first].push_back(QPointF(cpt, value));
        }

        if (m_fpsMaxAxis < stepData->m_totalMs){
            m_fpsMaxAxis = stepData->m_totalMs;
            updateAxis = true;
        }

        // keep max ms value
        if (m_maxFps < stepData->m_totalMs)
            m_maxFps = stepData->m_totalMs;

        cpt++;
    }

    m_series->replace(seriesPoints);
    m_selectionSeries->replace(selectedStepPoints);
    for (const auto& checkedSeries : m_checkedSeries)
    {
        auto it = checkedSeriesPoints.find(checkedSeries.first);
        if (it != checkedSeriesPoints.end())
        {
            checkedSeries.second.lineSeries->replace(it->second);
        }
    }

    // if needed enlarge the Y axis to cover new data
    if (updateAxis){
        m_axisY->setRange(0, m_fpsMaxAxis*1.1);
        m_chartView->updateYMax(m_fpsMaxAxis*1.1);
    }

    // every loop on buffer size check if Y axis can be reduced
    if ((m_step% m_bufferSize) == 0)
    {
        if (m_maxFps < m_fpsMaxAxis)
            m_fpsMaxAxis = m_maxFps;

        m_maxFps = 0;
        m_axisY->setRange(0, m_fpsMaxAxis*1.1);
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
    label_overheadValue->setText(QString::number(stepData->m_overheadMs));
    label_timersCounterValue->setText(QString::number(stepData->m_totalTimers));
}

void SofaWindowProfiler::updateTree(int step)
{
    const AnimationStepData* stepData = m_profilingData.at(step);

    tree_steps->setUpdatesEnabled(false);

    //clear old values
    tree_steps->clear();

    QList<QTreeWidgetItem*> root;
    root << addTreeItem(stepData);
    tree_steps->addTopLevelItems(root);

    //Expand the first two levels
    for (int i = 0; i < tree_steps->topLevelItemCount(); ++i)
    {
        auto* item = tree_steps->topLevelItem(i);
        item->setExpanded(true);
        for (int j = 0; j < item->childCount(); ++j)
        {
            item->child(i)->setExpanded(true);
        }
    }

    tree_steps->setUpdatesEnabled(true);
}

QTreeWidgetItem* SofaWindowProfiler::addTreeItem(AnimationSubStepData* subStep)
{
    QStringList columns;
    columns << QString::fromStdString(subStep->m_name);
    columns << QString::number(subStep->m_totalPercent, 'g', 2);
    columns << QString::number(subStep->m_selfPercent, 'g', 2);
    columns << QString::number(subStep->m_totalMs);
    columns << QString::number(subStep->m_selfMs);

    // add item to the tree
    QTreeWidgetItem* treeItem = new QTreeWidgetItem(columns);

    if (m_checkedSeries.find(subStep->m_name) == m_checkedSeries.end())
    {
        treeItem->setCheckState(0, Qt::Unchecked);
    }
    else
    {
        treeItem->setCheckState(0, Qt::Checked);
    }

    treeItem->setSelected(m_selectedStep == subStep->m_name);

    if (subStep->m_level <= 2)
    {
        QFont font = QApplication::font();
        font.setBold(true);

        for (int i = 0; i<treeItem->columnCount(); i++)
            treeItem->setFont(i, font);
    }

    QList<QTreeWidgetItem*> children;
    // process children
    for (auto* child : subStep->m_children)
        children << addTreeItem(child);
    treeItem->addChildren(children);

    return treeItem;
}

QTreeWidgetItem* SofaWindowProfiler::addTreeItem(const AnimationStepData* step)
{
    QStringList columns;
    columns << QString::fromStdString(step->m_idString);
    columns << "100";
    columns << QString::number(step->m_selfPercent, 'g', 2);
    columns << QString::number(step->m_totalMs);
    columns << QString::number(step->m_selfMs);

    QTreeWidgetItem* treeItem = new QTreeWidgetItem(columns);

    QFont font = QApplication::font();
    font.setBold(true);

    for (int i = 0; i<treeItem->columnCount(); i++)
        treeItem->setFont(i, font);

    // add new step items
    QList<QTreeWidgetItem*> children;
    for (auto* substep : step->m_subSteps)
    {
        children << addTreeItem(substep);
    }

    treeItem->addChildren(children);

    return treeItem;
}

void SofaWindowProfiler::onStepSelected(QTreeWidgetItem *item, int /*column*/)
{
    if (item->parent())
        m_selectedParentStep = item->parent()->text(0).toStdString();

    m_selectedStep = item->text(0).toStdString();

    if (item->checkState(0))
    {
        if (m_checkedSeries.find(m_selectedStep) == m_checkedSeries.end())
        {
            QLineSeries* checkedLineSeries = new QLineSeries;
            for (unsigned int i = 0; i < m_bufferSize; ++i)
            {
                const SReal value = (i < m_profilingData.size()) ?
                        m_profilingData[i]->getStepMs(m_selectedStep, m_selectedParentStep) : 0.f;
                checkedLineSeries->append(i, value);
            }

            m_chart->addSeries(checkedLineSeries);
            checkedLineSeries->setName(QString(m_selectedStep.c_str()));
            checkedLineSeries->attachAxis(m_axisY);

            m_checkedSeries.insert({m_selectedStep, {checkedLineSeries, m_selectedParentStep}});
        }
    }
    else
    {
        const auto it = m_checkedSeries.find(m_selectedStep);
        if (it != m_checkedSeries.end())
        {
            m_chart->removeSeries(it->second.lineSeries);
            m_checkedSeries.erase(it);
        }

        if (item->parent())
        {
            m_chart->addSeries(m_selectionSeries);
            int cpt = 0;
            QVector<QPointF> seriesPoints;
            for (auto* stepData : m_profilingData)
            {
                const SReal value = stepData->getStepMs(m_selectedStep, m_selectedParentStep);
                seriesPoints << QPointF(cpt++, value);
            }
            m_selectionSeries->replace(seriesPoints);
            m_selectionSeries->setName(QString(m_selectedStep.c_str()));
        }
    }
}

void SofaWindowProfiler::expandRootNodeOnly() const
{
    tree_steps->expandToDepth(0);
}
} //namespace sofa::gui::qt
