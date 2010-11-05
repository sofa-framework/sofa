/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SOFATUTORIALMANAGER_H
#define SOFA_SOFATUTORIALMANAGER_H

#include "TutorialSelector.h"
#include "GraphModeler.h"

#ifdef SOFA_QT4
#include <Q3MainWindow>
#include <QTextBrowser>
#include <QAction>
#include <QKeyEvent>
#include <QUrl>
#include <QComboBox>
#else
#include <qmainwindow.h>
#include <qtextbrowser.h>
#include <qaction.h>
#include <qurl.h>
#include <qpushbutton.h>
#include <qevent.h>
#include <qcombobox.h>
typedef QMainWindow Q3MainWindow;
#endif


namespace sofa
{

namespace gui
{

namespace qt
{



class SofaTutorialManager : public Q3MainWindow
{
    Q_OBJECT
public:
    SofaTutorialManager(QWidget* parent = 0, const char* name = 0);
    GraphModeler *getGraph() {return graph;}

    void keyPressEvent ( QKeyEvent * e );

public slots:
    void openCategory(const std::string &);
    void openTutorial(const std::string &filename);
    void openHTML(const std::string &filename);
    void launchScene();
    void editScene();
#ifdef SOFA_QT4
    void dynamicChangeOfScene( const QUrl&);
#else
    void dynamicChangeOfScene( const QString&);
#endif
signals:
    void runInSofa(const std::string& sceneFilename, GNode *root);
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
