#include "window.h"
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Window w;
    w.show();
    MainWindow windo;
    windo.show();
    return a.exec();
}
