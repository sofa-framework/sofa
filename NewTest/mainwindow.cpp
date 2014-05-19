#include "mainwindow.h"

MainWindow::MainWindow():QWidget()
{
    objectBox = new QCheckBox("Object",this);
    listeObject = new QComboBox();
    listeObject->addItem("cube");
    listeObject->addItem("sphere");
    listeObject->addItem("pyramid");
    lightBox = new QCheckBox("Number of light sources",this);
    numberLights = new QSpinBox();
    textureBox = new QCheckBox("Texture", this);
    listeTexture = new QComboBox();
    listeTexture->addItem("wood");
    listeTexture->addItem("checkerboard");
    m_button = new QPushButton("Apply", this);
    QObject::connect(m_button, SIGNAL(clicked()), this, SLOT(applyChoices()));

    layout = new QGridLayout;

    layout->addWidget(objectBox, 1, 0);
    layout->addWidget(listeObject, 1, 1);
    layout->addWidget(lightBox, 2, 0);
    layout->addWidget(numberLights, 2, 1);
    layout->addWidget(textureBox, 3, 0);
    layout->addWidget(listeTexture, 3, 1);
    layout->addWidget(m_button, 5, 0);

    this->setLayout(layout);
}
void MainWindow::applyChoices()
{
    QString objectToDraw;
    updateObject();
    updateLightsNumber();
    updateTexture();

}
void MainWindow::updateObject()
{
    if (objectBox->isChecked())
    {
        switch(listeObject->currentIndex())
        {
                case 0:objectToDraw = "cube";
                        QMessageBox::information(this, "objectToDraw", "Cube");
                break;
                case 1:objectToDraw = "sphere";
                        QMessageBox::information(this, "objectToDraw", "sphere");
                break;
                case 2:objectToDraw = "pyramid";
                        QMessageBox::information(this, "objectToDraw", "pyramid");
                break;
        }

    }
    else
    {
        objectToDraw = "cube";
        QMessageBox::information(this, "InfoElseObjectToDraw", "Cube");
    }
}
void MainWindow::updateLightsNumber()
{
    if (lightBox->isChecked())
    {
        lightsNumber = numberLights->value();
        QMessageBox::information(this, "lightsNumber", "lightsNumber = "+QString::number(lightsNumber));
    }
    else
    {
        lightsNumber = 1;
    }
}
void MainWindow::updateTexture()
{
    if (textureBox->isChecked())
    {
        switch(listeTexture->currentIndex())
        {
                case 0:
                    textureToDraw = "C:/Users/harid.TUMULTE/Documents/libQGL/NewTest/debug/bois.jpg";
                    QMessageBox::information(this, "textureToDraw", textureToDraw);
                    break;
                case 1:
                    textureToDraw = "C:/Users/harid.TUMULTE/Documents/libQGL/NewTest/debug/damier.jpg";
                    QMessageBox::information(this, "textureToDraw", textureToDraw);
                    break;
        }
    }
    else
    {
        textureToDraw = "C:/Users/harid.TUMULTE/Documents/libQGL/NewTest/debug/bois.jpg";
        QMessageBox::information(this, "textureToDraw", textureToDraw);
    }
}

MainWindow::~MainWindow()
{

}
