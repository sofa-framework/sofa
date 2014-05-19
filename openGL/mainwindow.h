#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QtGui/QMainWindow>
#include <QWidget>
#include <QGridLayout>
#include <QPushButton>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QComboBox>
#include <QSpinBox>
#include <QMessageBox>

#include <QtOpenGL>
#include <QGLWidget>

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    MainWindow();
    ~MainWindow();

    QCheckBox *objectBox;
    QCheckBox *lightBox;
    QCheckBox *textureBox;
    QGridLayout *layout;
    QPushButton *m_button;
    QComboBox *listeObject;
    QSpinBox *numberLights;
    QComboBox *listeTexture;
    QString objectToDraw;
    int lightsNumber;
    QString textureToDraw;

public slots:
    void applyChoices();
    void updateObject();
    void updateLightsNumber();
    void updateTexture();

};
#endif // MAINWINDOW_H
