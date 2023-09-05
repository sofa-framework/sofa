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
#include "GenGraphForm.h"

#ifdef WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#include <fstream>

#include <QFileDialog>
#include <QListView>
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <QHeaderView>
#include <QComboBox>
#include <QRadioButton>
#include <sofa/simulation/ExportDotVisitor.h>

namespace sofa::gui::qt
{

GenGraphForm::GenGraphForm(QWidget *parent)
    : QDialog(parent)
    , currentTask(nullptr)
    , settingFilter(false)
{
    setupUi(this);
    // signals and slots connections
    connect(browseButton, &QPushButton::clicked, this, &GenGraphForm::doBrowse);
    connect(exportButton, &QPushButton::clicked, this,  &GenGraphForm::doExport);
    connect(displayButton, &QPushButton::clicked, this,  &GenGraphForm::doDisplay);
    connect(closeButton, &QPushButton::clicked, this,  &GenGraphForm::doClose);
    connect(filename, &QLineEdit::textChanged, this, &GenGraphForm::change);
    connect(presetFilter, QOverload<int>::of(&QComboBox::activated), [=](int){ setFilter(); });
    connect(showNodes, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showObjects, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showBehaviorModels, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showCollisionModels, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showVisualModels, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showMappings, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showContext, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showCollisionPipeline, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showSolvers, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showMechanicalStates, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showForceFields, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showInteractionForceFields, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showConstraints, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showMass, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showTopology, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);
    connect(showMechanicalMappings, &QCheckBox::toggled, this, &GenGraphForm::changeFilter);

    // init preset filters
    {
        std::set<std::string>& filt = presetFilters["Full Graph"];
        filt.insert("showNodes");
        filt.insert("showObjects");
        filt.insert("showBehaviorModels");
        filt.insert("showCollisionModels");
        filt.insert("showVisualModels");
        filt.insert("showMappings");
        filt.insert("showContext");
        filt.insert("showCollisionPipeline");
        filt.insert("showSolvers");
        filt.insert("showMechanicalStates");
        filt.insert("showForceFields");
        filt.insert("showInteractionForceFields");
        filt.insert("showConstraints");
        filt.insert("showMass");
        filt.insert("showTopology");
        filt.insert("showMechanicalMappings");
    }
    {
        std::set<std::string>& filt = presetFilters["All Objects"];
        filt = presetFilters["Full Graph"];
        filt.erase("showNodes");
    }
    {
        std::set<std::string>& filt = presetFilters["All Nodes"];
        filt.insert("showNodes");
    }
    {
        std::set<std::string>& filt = presetFilters["Mechanical Graph"];
        filt = presetFilters["Full Graph"];
        filt.erase("showCollisionModels");
        filt.erase("showVisualModels");
        filt.erase("showMappings");
        filt.erase("showCollisionPipeline");
    }
    {
        std::set<std::string>& filt = presetFilters["Mechanical Objects"];
        filt = presetFilters["Mechanical Graph"];
        filt.erase("showNodes");
    }
    {
        std::set<std::string>& filt = presetFilters["Visual Graph"];
        filt.insert("showNodes");
        filt.insert("showObjects");
        filt.insert("showVisualModels");
    }
    {
        std::set<std::string>& filt = presetFilters["Visual Objects"];
        filt = presetFilters["Visual Graph"];
        filt.erase("showNodes");
    }
    {
        std::set<std::string>& filt = presetFilters["Collision Graph"];
        filt.insert("showNodes");
        filt.insert("showObjects");
        filt.insert("showCollisionModels");
        filt.insert("showCollisionPipeline");
    }
    {
        std::set<std::string>& filt = presetFilters["Collision Objects"];
        filt = presetFilters["Collision Graph"];
        filt.erase("showNodes");
    }
    {
        std::set<std::string>& filt = presetFilters["Collision Response Graph"];
        filt.insert("showNodes");
        filt.insert("showObjects");
        //filt.insert("showBehaviorModels");
        filt.insert("showCollisionModels");
        //filt.insert("showVisualModels");
        //filt.insert("showMappings");
        //filt.insert("showContext");
        //filt.insert("showCollisionPipeline");
        filt.insert("showSolvers");
        filt.insert("showMechanicalStates");
        //filt.insert("showForceFields");
        filt.insert("showInteractionForceFields");
        filt.insert("showConstraints");
        //filt.insert("showMass");
        //filt.insert("showTopology");
        filt.insert("showMechanicalMappings");
    }
    {
        std::set<std::string>& filt = presetFilters["Collision Response Objects"];
        filt = presetFilters["Collision Response Graph"];
        filt.erase("showNodes");
    }

}

void GenGraphForm::change()
{
    //exported = false;
    displayButton->setEnabled(false);
    if (exportButton->text() != "&Kill")
        exportButton->setEnabled(filename->text()!=QString(""));
}

void GenGraphForm::changeFilter()
{
    displayButton->setEnabled(false);
    if (settingFilter) return;
    const std::set<std::string> filt = getCurrentFilter();
    for (std::map<std::string,std::set<std::string> >::const_iterator it = presetFilters.begin(); it != presetFilters.end(); ++it)
    {
        if (it->second == filt)
        {
            const int index = presetFilter->findText(it->first.c_str());
            presetFilter->setCurrentIndex(index);
            return;
        }
    }
    const int index = presetFilter->findText("");
    presetFilter->setCurrentIndex(index);
}

QString removeFileExt(const QString &s)
{
    const int ext = s.lastIndexOf('.');
    if (ext==-1) return s;
    if (s.indexOf('/', ext+1) != -1|| s.indexOf('\\', ext+1) != -1)
        return s;
    return s.left(ext);
}

void GenGraphForm::doBrowse()
{
    QString s = QFileDialog::getSaveFileName(
            this,
            "Choose a file to save",
//            "export graph file dialog",
            filename->text(),
            "DOT Graph (*.dot)" );
    if (s.length()>0)
    {
        if (removeFileExt(s)==s) // no extension specified
            s += ".dot";
        filename->setText(s);
        change();
    }
}

void GenGraphForm::doExport()
{
    if (!tasks.empty())
    {
        killAllTasks();
        return;
    }
    if (filename->text()==QString("")) return;

    filename->setStyleSheet("");

    QString dotfile = filename->text();

    QString basefile = removeFileExt(dotfile);

    if (basefile==dotfile) // no extension specified
        dotfile += ".dot";

    std::ofstream fdot;
    fdot.open(dotfile.toStdString().c_str(), std::ofstream::out | std::ofstream::trunc);
    if (!fdot.is_open())
    {
        msg_error("GenGraphForm") << "Output to " << dotfile.toStdString() << " failed.";
        filename->clear();
        filename->setFocus();
        filename->setStyleSheet("border: 1px solid red");
        return;
    }
    {
        sofa::simulation::graph::ExportDotVisitor act(sofa::core::execparams::defaultInstance(), &fdot);
        act.showNode = this->showNodes->isChecked();
        act.showObject = this->showObjects->isChecked();
        act.showBehaviorModel = this->showBehaviorModels->isChecked();
        act.showCollisionModel = this->showCollisionModels->isChecked();
        act.showVisualModel = this->showVisualModels->isChecked();
        act.showMapping = this->showMappings->isChecked();
        act.showContext = this->showContext->isChecked();
        act.showCollisionPipeline = this->showCollisionPipeline->isChecked();
        act.showSolver = this->showSolvers->isChecked();
        act.showMechanicalState = this->showMechanicalStates->isChecked();
        act.showForceField = this->showForceFields->isChecked();
        act.showInteractionForceField = this->showInteractionForceFields->isChecked();
        act.showConstraint = this->showConstraints->isChecked();
        act.showMass = this->showMass->isChecked();
        act.showTopology = this->showTopology->isChecked();
        act.showMechanicalMapping = this->showMechanicalMappings->isChecked();
        act.labelNodeName = this->labelNodeName->isChecked();
        act.labelNodeClass = this->labelNodeClass->isChecked();
        act.labelObjectName = this->labelObjectName->isChecked();
        act.labelObjectClass = this->labelObjectClass->isChecked();
        graph->executeVisitor(&act);
    }
    fdot.close();
    msg_info("GenGraphForm") << "File saved in " << dotfile.toStdString();

    QStringList argv0;
    if (layoutDirV->isChecked())
        argv0 << "dot" << "-Grankdir=TB";
    else if (layoutDirH->isChecked())
        argv0 << "dot" << "-Grankdir=LR";
    else if (layoutSpring->isChecked())
        argv0 << "neato";
    else if (layoutRadial->isChecked())
        argv0 << "twopi";

    bool exp = false;

    exportedFile = dotfile;

    if (genPNG->isChecked())
    {
        QStringList argv = argv0;
        argv << "-Tpng" << "-o" << basefile+".png";
        argv << dotfile;
        addTask(argv);
        if (!exp)
        {
            exp = true;
            exportedFile = basefile+".png";
        }
    }

    if (genPS->isChecked())
    {
        QStringList argv = argv0;
        argv << "-Tps" << "-o" << basefile+".ps";
        argv << dotfile;
        addTask(argv);
        if (!exp)
        {
            exp = true;
            exportedFile = basefile+".ps";
        }
    }

    if (genFIG->isChecked())
    {
        QStringList argv = argv0;
        argv << "-Tfig" << "-o" << basefile+".fig";
        argv << dotfile;
        addTask(argv);
        if (!exp)
        {
            exp = true;
            exportedFile = basefile+".fig";
        }
    }

    if (genSVG->isChecked())
    {
        QStringList argv = argv0;
        argv << "-Tsvg" << "-o" << basefile+".svg";
        argv << dotfile;
        addTask(argv);
        if (!exp)
        {
            exp = true;
            exportedFile = basefile+".svg";
        }
    }
    //exported = true;
    //displayButton->setEnabled(true);
}

void GenGraphForm::doDisplay()
{
    if (exportedFile==QString("")) return;

    msg_info("GenGraphForm") << "OPEN " << exportedFile.toStdString();

#ifdef WIN32
    ShellExecuteA(nullptr, "open", exportedFile.toStdString().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    QStringList argv;
    argv << "display" << exportedFile;

    QString program = argv.front();
    argv.pop_front();
    //QProcess::startDetached(program, argv); //QString("start \"\"\"")+exportedFile+QString("\"\"\""));
    QProcess::startDetached(program, argv); //QString("start \"\"\"")+exportedFile+QString("\"\"\""));
#endif
}

void GenGraphForm::doClose()
{
    killAllTasks();
    this->close();
}

void GenGraphForm::setScene(sofa::simulation::Node* scene)
{
    graph = scene;
}

void GenGraphForm::killAllTasks()
{
    exportButton->setText("&Export");
    if (currentTask)
    {
        currentTask->kill();
        delete currentTask;
        currentTask = nullptr;
    }
    tasks.clear();
}

void GenGraphForm::addTask(QStringList argv)
{
    tasks.push_back(argv);
    if (currentTask == nullptr)
        runTask();
}

void GenGraphForm::taskFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    static const std::map<QProcess::ExitStatus, std::string> exitStatusMap {
        { QProcess::ExitStatus::NormalExit, "normal exit"},
        { QProcess::ExitStatus::CrashExit, "crash exit"},
    };
    const auto it = exitStatusMap.find(exitStatus);
    const auto exitStatusString = it == exitStatusMap.end() ? "unknown" : it->second;
    msg_info("GenGraphForm") << "TASK FINISHED (exit code: " << exitCode << ", exit status: " << exitStatusString << ")";
    if (currentTask == nullptr) return;

    delete currentTask;
    currentTask = nullptr;
    if (tasks.empty())
    {
        displayButton->setEnabled(true);
        exportButton->setText("&Export");
    }
    else
    {
        runTask();
    }
}

void GenGraphForm::taskError(QProcess::ProcessError error)
{
    static const std::map<QProcess::ProcessError, std::string> errorMap {
        { QProcess::ProcessError::FailedToStart, "failed to start"},
        { QProcess::ProcessError::Crashed, "crashed"},
        { QProcess::ProcessError::Timedout, "timed out"},
        { QProcess::ProcessError::ReadError, "read error"},
        { QProcess::ProcessError::WriteError, "write error"},
        { QProcess::ProcessError::UnknownError, "unknown error"},
    };
    const auto it = errorMap.find(error);
    const auto errorString = it == errorMap.end() ? "unknown" : it->second;
    msg_error("GenGraphForm") << "An error occured: " << errorString;

    if (error == QProcess::ProcessError::FailedToStart)
    {
        msg_error("GenGraphForm") << "Check that the required program (Graphviz) is in your PATH environment variable";
    }

    exportButton->setText("&Export");
    if (currentTask)
    {
        currentTask->kill();
        delete currentTask;
        currentTask = nullptr;
    }
}

void GenGraphForm::runTask()
{
    QStringList argv = tasks.front();
    tasks.pop_front();
    exportButton->setText("&Kill");
    const QString cmd = argv.join(QString(" "));
    msg_info("GenGraphForm") << "STARTING TASK " << cmd.toStdString();

    auto* p = new QProcess(this);
    const QString program = argv.front();
    argv.pop_front();
#if (QT_VERSION < QT_VERSION_CHECK(5, 13, 0))
    p->setReadChannelMode(QProcess::ForwardedChannels);
#else
    p->setProcessChannelMode(QProcess::ForwardedChannels);
#endif
    connect(p,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this,&GenGraphForm::taskFinished);
    connect(p,&QProcess::errorOccurred,this,&GenGraphForm::taskError);
    p->start(program, argv);
    currentTask = p;
}

void GenGraphForm::setFilter()
{
    const QString fname = presetFilter->currentText();
    if (fname == "") return;
    const std::string name = fname.toStdString();
    if (presetFilters.count(name) > 0)
    {
        const std::set<std::string> & filt = presetFilters[name];
        settingFilter = true;
        this->showNodes->setChecked(filt.find("showNodes")!=filt.end());
        this->showObjects->setChecked(filt.find("showObjects")!=filt.end());
        this->showBehaviorModels->setChecked(filt.find("showBehaviorModels")!=filt.end());
        this->showCollisionModels->setChecked(filt.find("showCollisionModels")!=filt.end());
        this->showVisualModels->setChecked(filt.find("showVisualModels")!=filt.end());
        this->showMappings->setChecked(filt.find("showMappings")!=filt.end());
        this->showContext->setChecked(filt.find("showContext")!=filt.end());
        this->showCollisionPipeline->setChecked(filt.find("showCollisionPipeline")!=filt.end());
        this->showSolvers->setChecked(filt.find("showSolvers")!=filt.end());
        this->showMechanicalStates->setChecked(filt.find("showMechanicalStates")!=filt.end());
        this->showForceFields->setChecked(filt.find("showForceFields")!=filt.end());
        this->showInteractionForceFields->setChecked(filt.find("showInteractionForceFields")!=filt.end());
        this->showConstraints->setChecked(filt.find("showConstraints")!=filt.end());
        this->showMass->setChecked(filt.find("showMass")!=filt.end());
        this->showTopology->setChecked(filt.find("showTopology")!=filt.end());
        this->showMechanicalMappings->setChecked(filt.find("showMechanicalMappings")!=filt.end());
        settingFilter = false;
        displayButton->setEnabled(false);
        const int index = presetFilter->findText(fname);
        presetFilter->setCurrentIndex(index);
    }
    else
    {
        //if (QMessage::question(this,"Graph Preset Filters",...)
        presetFilters[name] = getCurrentFilter();
    }
}

std::set<std::string> GenGraphForm::getCurrentFilter()
{
    std::set<std::string> filt;
    if (this->showNodes->isChecked()) filt.insert("showNodes");
    if (this->showObjects->isChecked()) filt.insert("showObjects");
    if (this->showBehaviorModels->isChecked()) filt.insert("showBehaviorModels");
    if (this->showCollisionModels->isChecked()) filt.insert("showCollisionModels");
    if (this->showVisualModels->isChecked()) filt.insert("showVisualModels");
    if (this->showMappings->isChecked()) filt.insert("showMappings");
    if (this->showContext->isChecked()) filt.insert("showContext");
    if (this->showCollisionPipeline->isChecked()) filt.insert("showCollisionPipeline");
    if (this->showSolvers->isChecked()) filt.insert("showSolvers");
    if (this->showMechanicalStates->isChecked()) filt.insert("showMechanicalStates");
    if (this->showForceFields->isChecked()) filt.insert("showForceFields");
    if (this->showInteractionForceFields->isChecked()) filt.insert("showInteractionForceFields");
    if (this->showConstraints->isChecked()) filt.insert("showConstraints");
    if (this->showMass->isChecked()) filt.insert("showMass");
    if (this->showTopology->isChecked()) filt.insert("showTopology");
    if (this->showMechanicalMappings->isChecked()) filt.insert("showMechanicalMappings");
    return filt;
}

} //namespace sofa::gui::qt
