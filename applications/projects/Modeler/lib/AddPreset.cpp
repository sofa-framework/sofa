/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "AddPreset.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <sofa/gui/qt/FileManagement.h> //static functions to manage opening/ saving of files
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpacerItem>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>

namespace sofa
{

namespace gui
{

namespace qt
{


AddPreset::AddPreset(QWidget* parent):
    QDialog(parent)
{
    QGridLayout * fileFormGridLayout = new QGridLayout();
    openFileText0 = new QLabel("Path to the Mesh File", this);
    fileFormGridLayout->addWidget(openFileText0, 0, 0);
    openFilePath0 = new QLineEdit(this);
    fileFormGridLayout->addWidget(openFilePath0, 0, 1);
    openFileButton0 = new QPushButton("Browse", this);
    fileFormGridLayout->addWidget(openFileButton0, 0, 2);
    connect(openFileButton0, SIGNAL(clicked()), this, SLOT(fileOpen()));

    fileFormGridLayout->setSpacing(6);
    openFileText1 = new QLabel("Path to the VisualModel", this);
    fileFormGridLayout->addWidget(openFileText1, 1, 0);
    openFilePath1 = new QLineEdit(this);
    fileFormGridLayout->addWidget(openFilePath1, 1, 1);
    openFileButton1 = new QPushButton("Browse", this);
    fileFormGridLayout->addWidget(openFileButton1, 1, 2);
    connect(openFileButton1, SIGNAL(clicked()), this, SLOT(fileOpen()));

    fileFormGridLayout->setSpacing(6);
    openFileText2 = new QLabel("Path to the CollisionModel", this);
    fileFormGridLayout->addWidget(openFileText2, 2, 0);
    openFilePath2 = new QLineEdit(this);
    fileFormGridLayout->addWidget(openFilePath2, 2, 1);
    openFileButton2 = new QPushButton("Browse", this);
    fileFormGridLayout->addWidget(openFileButton2, 2, 2);
    connect(openFileButton2, SIGNAL(clicked()), this, SLOT(fileOpen()));

    QGridLayout * gridLayoutTwo = new QGridLayout();
    gridLayoutTwo->addWidget(new QLabel("Initial Position", this), 0, 0);
    positionX = new QLineEdit(this);
    gridLayoutTwo->addWidget(positionX, 0, 1);
    positionY = new QLineEdit(this);
    gridLayoutTwo->addWidget(positionY, 0, 2);
    positionZ = new QLineEdit(this);
    gridLayoutTwo->addWidget(positionZ, 0, 3);

    gridLayoutTwo->addWidget(new QLabel("Initial Rotation", this), 1, 0);
    rotationX = new QLineEdit(this);
    gridLayoutTwo->addWidget(rotationX, 1, 1);
    rotationY = new QLineEdit(this);
    gridLayoutTwo->addWidget(rotationY, 1, 2);
    rotationZ = new QLineEdit(this);
    gridLayoutTwo->addWidget(rotationZ, 1, 3);

    gridLayoutTwo->addWidget(new QLabel("Initial Scale", this), 2, 0);
    scaleX = new QLineEdit(this);
    gridLayoutTwo->addWidget(scaleX, 2, 0);
    scaleY = new QLineEdit(this);
    gridLayoutTwo->addWidget(scaleY, 2, 0);
    scaleZ = new QLineEdit(this);
    gridLayoutTwo->addWidget(scaleZ, 2, 0);

    QHBoxLayout *buttonHLayout = new QHBoxLayout();
    buttonHLayout->addItem(new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum));
    buttonOk = new QPushButton("&OK", this);
    buttonOk->setDefault(true);
    buttonHLayout->addWidget(buttonOk);
    connect(buttonOk, SIGNAL(clicked()), this, SLOT(accept()));

    buttonCancel = new QPushButton("&Cancel", this);
    buttonHLayout->addWidget(buttonCancel);
    connect(buttonCancel, SIGNAL(clicked()), this, SLOT(reject()));

    QVBoxLayout *verticalLayout = new QVBoxLayout(this);
    verticalLayout->addLayout(fileFormGridLayout);
    verticalLayout->addLayout(gridLayoutTwo);
    verticalLayout->addLayout(buttonHLayout);

    setWindowTitle("Add a scene or an object");
    clear();

    //Make the connection between this widget and the parent
    connect(this, SIGNAL(loadPreset(Node*,std::string,std::string*, std::string,std::string,std::string)),
            parent, SLOT(loadPreset(Node*,std::string,std::string*, std::string,std::string,std::string)));
}

void AddPreset::setElementPresent(bool *elementPresent)
{
    if (elementPresent != NULL)
    {
        if (!elementPresent[0])
        {
            openFileText0->hide();
            openFilePath0->hide();
            openFileButton0->hide();
        }
        else
        {
            openFileText0->show();
            openFilePath0->show();
            openFileButton0->show();
        }
        if (!elementPresent[1])
        {
            openFileText1->hide();
            openFilePath1->hide();
            openFileButton1->hide();
        }
        else
        {
            openFileText1->show();
            openFilePath1->show();
            openFileButton1->show();
        }
        if (!elementPresent[2])
        {
            openFileText2->hide();
            openFilePath2->hide();
            openFileButton2->hide();
        }
        else
        {
            openFileText2->show();
            openFilePath2->show();
            openFileButton2->show();
        }
    }
}
//Clear the dialoag
void AddPreset::clear()
{
    positionX->setText("0.0");
    positionY->setText("0.0");
    positionZ->setText("0.0");

    rotationX->setText("0.0");
    rotationY->setText("0.0");
    rotationZ->setText("0.0");

    scaleX->setText("1.0");
    scaleY->setText("1.0");
    scaleZ->setText("1.0");

    openFilePath0->setText(NULL);
    openFilePath1->setText(NULL);
    openFilePath2->setText(NULL);
}

// When the Ok Button is clicked, this method is called: we just have
// to emit a signal to the parent, with the information on the object
void AddPreset::accept()
{
    std::string position;
    std::string rotation;
    std::string scale;

    std::string filenames[3];
    //In case of static objects
    if (openFileText0->isVisible())
        filenames[0] = openFilePath0->text().toStdString();
    else
        filenames[0]=openFilePath2->text().toStdString();

    filenames[1] = openFilePath1->text().toStdString();
    filenames[2] = openFilePath2->text().toStdString();

    std::ostringstream out;
    out << positionX->text().toStdString()<<" "<<positionY->text().toStdString()<<" "<<positionZ->text().toStdString();
    position=out.str();
    out.str("");
    out << rotationX->text().toStdString()<<" "<<rotationY->text().toStdString()<<" "<<rotationZ->text().toStdString();
    rotation=out.str();
    out.str("");
    out << scaleX->text().toStdString()<<" "<<scaleY->text().toStdString()<<" "<<scaleZ->text().toStdString();
    scale=out.str();
    out.str("");

    emit( loadPreset(node, presetFile,filenames,position,rotation,scale));
    clear();
    QDialog::accept();
}


//**************************************************************************************
//Open a file Dialog and set the path of the selected path in the text field.
void AddPreset::fileOpen()
{
    if (sofa::helper::system::DataRepository.findFile(fileName))
        fileName=sofa::helper::system::DataRepository.getFile(fileName);
    else
        fileName=sofa::helper::system::DataRepository.getFirstPath();

    QString s  = getOpenFileName(this, QString(fileName.c_str()), "Mesh File (*.obj );;All (*)", "open file dialog",  "Choose a file to open" );
    const std::string SofaPath (sofa::helper::system::DataRepository.getFirstPath().c_str());

    if (s.isNull() ) return;
    fileName=std::string (s.toStdString());

    std::string filePath = sofa::helper::system::FileRepository::relativeToPath(fileName,SofaPath);
    if( filePath == fileName)
    {
        if (!relative.empty())
        {
            //size_t loc = fileName.find( relative, 0 );
            fileName = fileName.substr(relative.size()+1);
        }
    }

    if (sender() == openFileButton0)
    {
        openFilePath0->setText(QString(filePath.c_str()));
    }
    else if (sender() == openFileButton1)
    {
        openFilePath1->setText(QString(filePath.c_str()));
    }
    else if (sender() == openFileButton2)
    {
        openFilePath2->setText(QString(filePath.c_str()));
    }
}

} // namespace qt

} // namespace gui

} // namespace sofa

