#include <QtGui/QApplication>
#include "mywindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    myWindow w;
    w.show();
    return a.exec();
}
