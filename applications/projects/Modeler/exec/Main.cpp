#include <iostream>
#include <fstream>

#include <qapplication.h>
#include "../lib/SofaModeler.h"
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{

    QApplication* application = new QApplication(argc, argv);

    sofa::gui::qt::SofaModeler* sofaModeler = new sofa::gui::qt::SofaModeler();
    application->setMainWidget(sofaModeler);
    sofaModeler->show();
    return application->exec();
}
