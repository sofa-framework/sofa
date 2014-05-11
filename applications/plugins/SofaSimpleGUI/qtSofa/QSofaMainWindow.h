#ifndef QTSOFAMAINWINDOW_H
#define QTSOFAMAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <string>
using std::string;
#include <vector>
using std::vector;
#include "QSofaScene.h"
class QSofaViewer;

/**
 * @brief The QSofaMainWindow class contains a Sofa simulation.
 *
 * @author Francois Faure, 2014
 */
class QSofaMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit QSofaMainWindow(QWidget *parent = 0);

    /**
     * @brief initSofa
     * @param plugins list of plugin (names) to load
     * @param filename scene to load on startup
     */
    void initSofa(const std::vector<string> &plugins, string filename );

    void start();

signals:

public slots:

protected:
    sofa::newgui::QSofaScene sofaScene;
    QSofaViewer* sofaViewer1;
    QTimer *timer;

};

#endif // QTSOFAMAINWINDOW_H
