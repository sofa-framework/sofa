#include <QApplication>
#include "myWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    myWindow myWin;
    myWin.show();
    return app.exec();
}
