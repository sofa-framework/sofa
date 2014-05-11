#include "QSofaMainWindow.h"
#include "QSofaViewer.h"
#include <QMessageBox>


QSofaMainWindow::QSofaMainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    sofaViewer1 = new QSofaViewer(&sofaScene,this);
    setCentralWidget(sofaViewer1);

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), &sofaScene, SLOT(step()));

}

void QSofaMainWindow::initSofa( const std::vector<std::string> &plugins, string fileName )
{
    // --- Init sofa ---
    sofaScene.init(plugins,fileName);
    QMessageBox::information( this, tr("Tip"), tr("Shift-Click and drag the control points to interact.\nRelease button before Shift to release the control point.\nRelease Shift before button to keep it attached where it is.") );

}

void QSofaMainWindow::start()
{
    timer->start(40);
}
