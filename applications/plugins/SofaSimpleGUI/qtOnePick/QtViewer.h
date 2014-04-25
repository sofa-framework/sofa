#ifndef QTVIEWER_H
#define QTVIEWER_H

#include <QGLWidget>

class QtViewer : public QGLWidget
{
    Q_OBJECT
public:
    explicit QtViewer(QGLWidget *parent = 0);
    void initializeGL();
    void paintGL();
    void resizeGL(int w, int h);


signals:

public slots:

};

#endif // QTVIEWER_H
