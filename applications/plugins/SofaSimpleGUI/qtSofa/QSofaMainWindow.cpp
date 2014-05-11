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
using std::cout;
using std::endl;


QSofaMainWindow::QSofaMainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    setAcceptDrops(true);

    sofaViewer1 = new QSofaViewer(&sofaScene,this);
    setCentralWidget(sofaViewer1);

    QToolBar* toolbar = addToolBar(tr("Controls"));
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    QMenu* simulationMenu = menuBar()->addMenu(tr("&Simulation"));

    // start/stop
    startAct = new QAction(QIcon(":/icons/start.svg"), tr("&Play..."), this);
    startAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPlay));
    startAct->setShortcut(QKeySequence(Qt::Key_Space));
    startAct->setStatusTip(tr("Start simulation"));
    connect(startAct, SIGNAL(triggered()), &sofaScene, SLOT(playpause()));
    connect(&sofaScene, SIGNAL(sigPlaying(bool)), this, SLOT(isPlaying(bool)) );
    this->addAction(startAct);
    simulationMenu->addAction(startAct);
    toolbar->addAction(startAct);

    // reset
    QAction* resetAct = new QAction(QIcon(":/icons/reset.svg"), tr("&Reset..."), this);
    resetAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaSkipBackward));
    resetAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_R));
    resetAct->setStatusTip(tr("Reset simulation"));
    connect(resetAct, SIGNAL(triggered()), &sofaScene, SLOT(reset()));
    this->addAction(resetAct);
    simulationMenu->addAction(resetAct);
    toolbar->addAction(resetAct);

    // open
    QAction* openAct = new QAction(QIcon(":/icons/reset.svg"), tr("&Open..."), this);
    openAct->setIcon(this->style()->standardIcon(QStyle::SP_FileIcon));
    openAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_O));
    openAct->setStatusTip(tr("Open simulation"));
    connect(openAct, SIGNAL(triggered()), this, SLOT(open()));
    this->addAction(openAct);
    fileMenu->addAction(openAct);
    toolbar->addAction(openAct);

    // reload
    QAction* reloadAct = new QAction(QIcon(":/icons/reload.svg"), tr("&Reload..."), this);
    reloadAct->setIcon(this->style()->standardIcon(QStyle::SP_BrowserReload));
    reloadAct->setShortcut(QKeySequence(Qt::CTRL+Qt::SHIFT+Qt::Key_R));
    reloadAct->setStatusTip(tr("Reload simulation"));
    connect(reloadAct, SIGNAL(triggered()), &sofaScene, SLOT(reload()));
    this->addAction(reloadAct);
    fileMenu->addAction(reloadAct);
    toolbar->addAction(reloadAct);

}

void QSofaMainWindow::initSofa( const std::vector<std::string> &plugins, string fileName )
{
    // --- Init sofa ---
    sofaScene.init(plugins,fileName);
    QMessageBox::information( this, tr("Tip"), tr("Space to start/stop,\n\n"
                                                  "Shift-Click and drag the control points to interact.\n"
                                                  "Release button before Shift to release the control point.\n"
                                                  "Release Shift before button to keep it attached where it is.") );

}

void QSofaMainWindow::isPlaying( bool playing )
{
    if( playing ) // propose to pause
        startAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPause));
    else // propose to play
        startAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPlay));
}

void QSofaMainWindow::open()
{
    sofaScene.pause();
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open scene file"), ".", tr("Scene Files (*.scn *.xml *.py)"));
    sofaScene.open(fileName.toStdString().c_str());
}

