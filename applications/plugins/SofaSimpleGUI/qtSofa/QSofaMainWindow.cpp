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
using std::cout;
using std::endl;


QSofaMainWindow::QSofaMainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    setFocusPolicy(Qt::ClickFocus);

    sofaViewer1 = new QSofaViewer(&sofaScene,this);
    setCentralWidget(sofaViewer1);

    QToolBar* toolbar = addToolBar(tr("Controls"));
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    QMenu* simulationMenu = menuBar()->addMenu(tr("&Simulation"));

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
    spinBox->setMaxValue(40000);
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
    connect(reloadAct, SIGNAL(triggered()), &sofaScene, SLOT(reload()));
    this->addAction(reloadAct);
    fileMenu->addAction(reloadAct);
    toolbar->addAction(reloadAct);
    }

    // viewAll
    {
    QAction* viewAllAct = new QAction(QIcon(":/icons/eye.svg"), tr("&ViewAll..."), this);
    viewAllAct->setShortcut(QKeySequence(Qt::CTRL+Qt::SHIFT+Qt::Key_V));
    viewAllAct->setToolTip(tr("Adjust camera to view all"));
    connect(viewAllAct, SIGNAL(triggered()), sofaViewer1, SLOT(viewAll()));
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

    //=======================================
    sofaViewer1->setFocus();

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
        _playPauseAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPause));
    else // propose to play
        _playPauseAct->setIcon(this->style()->standardIcon(QStyle::SP_MediaPlay));
}

void QSofaMainWindow::open()
{
    sofaScene.pause();
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open scene file"), ".", tr("Scene Files (*.scn *.xml *.py)"));
    if( fileName.size()>0 )
        sofaScene.open(fileName.toStdString().c_str());
}

void QSofaMainWindow::setDt( int milis ) { sofaScene.setTimeStep( milis/1000.0 ); }

