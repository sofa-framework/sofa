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
#include "QSofaMainWindow.h"
#include "QSofaViewer.h"
#include <QMessageBox>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QToolBar>
#include <QStyle>
#include <QFileDialog>
#include <QString>
#include <iostream>
#include <QSpinBox>
#include <QDockWidget>
#include "oneTetra.h"
using std::cout;
using std::endl;


QSofaMainWindow::QSofaMainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    setFocusPolicy(Qt::ClickFocus);

    mainViewer = new QSofaViewer(&sofaScene,NULL,this);
    setCentralWidget(mainViewer);

    QToolBar* toolbar = addToolBar(tr("Controls"));
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    QMenu* simulationMenu = menuBar()->addMenu(tr("&Simulation"));
    QMenu* viewMenu = menuBar()->addMenu(tr("&View"));

    // find icons at https://www.iconfinder.com/search

    // start/stop
    {
        _playPauseAct = new QAction(QIcon(":/icons/start.svg"), tr("&Play..."), this);
        _playPauseAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPlay));
        _playPauseAct->setShortcut(QKeySequence(Qt::Key_Space));
        _playPauseAct->setToolTip(tr("Play/Pause simulation"));
        connect(_playPauseAct, SIGNAL(triggered()), &sofaScene, SLOT(playpause()));
        connect(&sofaScene, SIGNAL(sigPlaying(bool)), this, SLOT(isPlaying(bool)) );
        this->addAction(_playPauseAct);
        simulationMenu->addAction(_playPauseAct);
        toolbar->addAction(_playPauseAct);
    }

    // reset
    {
        QAction* resetAct = new QAction(QIcon(":/icons/reset.svg"), tr("&Reset..."), this);
        resetAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaSkipBackward));
        resetAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_R));
        resetAct->setToolTip(tr("Restart from the beginning, without reloading"));
        connect(resetAct, SIGNAL(triggered()), &sofaScene, SLOT(reset()));
        this->addAction(resetAct);
        simulationMenu->addAction(resetAct);
        toolbar->addAction(resetAct);
    }

    // open
    {
        QAction* openAct = new QAction(QIcon(":/icons/reset.svg"), tr("&Open..."), this);
        openAct->setIcon(this->style()->standardIcon(QStyle::SP_FileIcon));
        openAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_O));
        openAct->setToolTip(tr("Open new scene"));
        openAct->setStatusTip(tr("Opening scene…"));
        connect(openAct, SIGNAL(triggered()), this, SLOT(open()));
        this->addAction(openAct);
        fileMenu->addAction(openAct);
        toolbar->addAction(openAct);
    }

    // time step
    {
        QSpinBox* spinBox = new QSpinBox(this);
        toolbar->addWidget(spinBox);
        spinBox->setValue(40);
        spinBox->setMaximum(40000);
        spinBox->setToolTip(tr("Simulation time step (ms)"));
        connect(spinBox,SIGNAL(valueChanged(int)), this, SLOT(setDt(int)));
    }

    // reload
    {
        QAction* reloadAct = new QAction(QIcon(":/icons/reload.svg"), tr("&Reload..."), this);
        reloadAct->setIcon(this->style()->standardIcon(QStyle::SP_BrowserReload));
        reloadAct->setShortcut(QKeySequence(Qt::CTRL+Qt::SHIFT+Qt::Key_R));
        reloadAct->setStatusTip(tr("Reloading scene…"));
        reloadAct->setToolTip(tr("Reload file and restart from the beginning"));
        connect(reloadAct, SIGNAL(triggered()), this, SLOT(reload()));
        this->addAction(reloadAct);
        fileMenu->addAction(reloadAct);
//        toolbar->addAction(reloadAct);
    }

    // viewAll
    {
        QAction* viewAllAct = new QAction(QIcon(":/icons/eye.svg"), tr("&ViewAll..."), this);
        viewAllAct->setShortcut(QKeySequence(Qt::CTRL+Qt::SHIFT+Qt::Key_V));
        viewAllAct->setToolTip(tr("Adjust camera to view all"));
        connect(viewAllAct, SIGNAL(triggered()), mainViewer, SLOT(viewAll()));
        this->addAction(viewAllAct);
        simulationMenu->addAction(viewAllAct);
        toolbar->addAction(viewAllAct);
    }
	
    // print
    {
        QAction* printAct = new QAction( QIcon(":/icons/print.svg"), tr("&PrintGraph..."), this);
        printAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_P));
        printAct->setToolTip(tr("Print the graph on the standard output"));
        connect(printAct, SIGNAL(triggered()), &sofaScene, SLOT(printGraph()));
        this->addAction(printAct);
        simulationMenu->addAction(printAct);
        toolbar->addAction(printAct);
    }

//    {
//        QAction* toggleFullScreenAct = new QAction( tr("&FullScreen"), this );
//        toggleFullScreenAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_F));
//        toggleFullScreenAct->setToolTip(tr("Show full screen"));
//        connect(toggleFullScreenAct, SIGNAL(triggered()), this, SLOT(toggleFullScreen()));
//        this->addAction(toggleFullScreenAct);
//        viewMenu->addAction(toggleFullScreenAct);
//        toolbar->addAction(toggleFullScreenAct);
//        _fullScreen = false;
//    }

    {
        QAction* createAdditionalViewerAct = new QAction( tr("&Additional viewer"), this );
        createAdditionalViewerAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_V));
        createAdditionalViewerAct->setToolTip(tr("Add/remove additional viewer"));
        connect(createAdditionalViewerAct, SIGNAL(triggered()), this, SLOT(createAdditionalViewer()));
        this->addAction(createAdditionalViewerAct);
        viewMenu->addAction(createAdditionalViewerAct);
        toolbar->addAction(createAdditionalViewerAct);
    }

    //=======================================
    mainViewer->setFocus();

}

void QSofaMainWindow::initSofa(string fileName )
{
    // --- Init sofa ---
    if(fileName.empty())
    {
        cout << "no fileName provided, using default scene" << endl;
		oneTetra();
        //sofaScene.setScene(oneTetra()); // the sofa::Simulation is a singleton, the call to oneTetra already loaded the scene
    }
    else {
        sofaScene.open(fileName.c_str());
    }
    QMessageBox::information( this, tr("Tip"), tr("Space to start/stop,\n\n"
                                                  "Shift-Click and drag the control points to interact. Use Ctrl-Shift-Click to select Interactors only\n"
                                                  "Release button before Shift to release the control point.\n"
                                                  "Release Shift before button to keep it attached where it is.") );

}

void QSofaMainWindow::isPlaying( bool playing )
{
    if( playing ) // propose to pause
        _playPauseAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPause));
    else // propose to play
        _playPauseAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPlay));
}

void QSofaMainWindow::open()
{
    sofaScene.pause();
    std::string path = std::string(QTSOFA_SRC_DIR) + "/../examples";
    _fileName = QFileDialog::getOpenFileName(this, tr("Open scene file"), path.c_str(), tr("Scene Files (*.scn *.xml *.py)"));
    if( _fileName.size()>0 )
        sofaScene.open(_fileName.toStdString().c_str());
}

void QSofaMainWindow::reload() {
    if( _fileName.size()==0 )
    {
        QMessageBox::information( this, tr("Error"), tr("No file to reload") );
        return;
    }
    sofaScene.open(_fileName.toStdString().c_str());
}


void QSofaMainWindow::setDt( int milis ) { sofaScene.setTimeStep( milis/1000.0 ); }

void QSofaMainWindow::toggleFullScreen()
{
    _fullScreen = !_fullScreen;
    if( _fullScreen ){
        this->showFullScreen();
    }
    else {
        this->showNormal();
    }
}

void QSofaMainWindow::createAdditionalViewer()
{
    QSofaViewer* additionalViewer = new QSofaViewer(&sofaScene, mainViewer, this);
    QDockWidget* additionalViewerDock = new QDockWidget(tr("Additional Viewer"), this);
    additionalViewerDock->setWidget(additionalViewer);
    addDockWidget(Qt::LeftDockWidgetArea, additionalViewerDock);
}
