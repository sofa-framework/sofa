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
#include <ui_BaseGenGraphForm.h>

#include "QProcess"
#include "QStringList"
#include <sofa/simulation/Node.h>

#include <list>
#include <map>
#include <set>

namespace sofa::gui::qt
{

class GenGraphForm : public QDialog, public Ui_BaseGenGraphForm
{
    Q_OBJECT
public:
    GenGraphForm(QWidget *parent = Q_NULLPTR);

    void setScene(sofa::simulation::Node* scene);

public slots:
    virtual void change();
    virtual void doBrowse();
    virtual void doExport();
    virtual void doDisplay();
    virtual void doClose();
    virtual void taskFinished(int exitCode ,QProcess::ExitStatus exitStatus);
    virtual void taskError(QProcess::ProcessError error);
    virtual void changeFilter();
    virtual void setFilter();

protected:
    QString exportedFile;
    sofa::simulation::Node* graph;
    std::list<QStringList> tasks;
    QProcess* currentTask;

    void addTask(QStringList argv);
    void runTask();
    void killAllTasks();

    std::map<std::string, std::set<std::string> > presetFilters;
    bool settingFilter;
    std::set<std::string> getCurrentFilter();
};

} //namespace sofa::gui::qt
