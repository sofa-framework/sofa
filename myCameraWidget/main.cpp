#include <QtGui/QApplication>
#include "mycamerawidget.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    myCameraWidget w;
    w.show();
    return a.exec();
}
