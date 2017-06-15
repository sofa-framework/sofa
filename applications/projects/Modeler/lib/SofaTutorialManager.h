/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SOFATUTORIALMANAGER_H
#define SOFA_SOFATUTORIALMANAGER_H

#include "TutorialSelector.h"
#include "GraphModeler.h"

#include <QMainWindow>
#include <QTextBrowser>
#include <QAction>
#include <QKeyEvent>
#include <QUrl>
#include <QComboBox>


namespace sofa
{

namespace gui
{

namespace qt
{



class SofaTutorialManager : public QMainWindow
{
    Q_OBJECT
public:
    SofaTutorialManager(QWidget* parent = 0, const char *name = "");
    GraphModeler *getGraph() {return graph;}

    void keyPressEvent ( QKeyEvent * e );

public slots:
    void openCategory(const std::string &);
    void openTutorial(const std::string &filename);
    void openHTML(const std::string &filename);
    void launchScene();
    void editScene();
    void dynamicChangeOfScene( const QUrl&);
signals:
    void runInSofa(const std::string& sceneFilename, Node *root);
    void undo();
    void redo();
    void editInModeler(const std::string& sceneFilename);

protected:
    TutorialSelector *selector;
    GraphModeler *graph;
    QTextBrowser* descriptionPage;
    QPushButton *buttonRunInSofa;
    QComboBox *tutorialList;
    QPushButton* buttonEditInModeler;
};

}
}
}

#endif
