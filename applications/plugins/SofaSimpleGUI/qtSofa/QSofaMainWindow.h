#ifndef QTSOFAMAINWINDOW_H
#define QTSOFAMAINWINDOW_H

#include <QMainWindow>
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

    sofa::newgui::QSofaScene sofaScene;
    QSofaViewer* sofaViewer1;

signals:

public slots:
    /**
     * @brief used to change the play/pause icon
     */
    void isPlaying(bool);
    /**
     * @brief Select a new scene file using the menu, clear the current scene and replace it with the new one
     */
    void open();
    void setDt( int ms );
protected:
    QAction* startAct;
};

#endif // QTSOFAMAINWINDOW_H
