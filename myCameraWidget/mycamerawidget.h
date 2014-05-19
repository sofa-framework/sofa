#ifndef MYCAMERAWIDGET_H
#define MYCAMERAWIDGET_H

#include <QtGui/QMainWindow>
#include <QtOpenGL>
#include <QGLWidget>
class myCameraWidget : public QMainWindow
{
    Q_OBJECT

public:
    myCameraWidget(QWidget *parent = 0);
    ~myCameraWidget();
};

#endif // MYCAMERAWIDGET_H
