/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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

#ifdef SOFA_QT4
#include <Q3FileDialog>
#include <QLineEdit>
#include <QLabel>
#include <QRadioButton>
#include <QPushButton>
#include <Q3ButtonGroup>
#include <QGridLayout>
#else
#include <qfiledialog.h>
#include <qlineedit.h>
#include <qlabel.h>
#include <qradiobutton.h>
#include <qpushbutton.h>
#include <qbuttongroup.h>
#include <qlayout.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{


#ifndef SOFA_QT4
typedef QFileDialog  Q3FileDialog;
typedef QButtonGroup Q3ButtonGroup;
#endif


AddPreset::AddPreset(  QWidget* parent , const char* name,bool , Qt::WFlags ):	DialogAddPreset(parent, name)
{
    this->setCaption(QString(sofa::helper::system::SetDirectory::GetFileName(name).c_str()));
    clear();

    //Make the connection between this widget and the parent
    connect( this, SIGNAL(loadPreset(GNode*,std::string,std::string*, std::string,std::string,std::string)),
            parent, SLOT(loadPreset(GNode*,std::string,std::string*, std::string,std::string,std::string)));
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
    positionX->setText("0");
    positionY->setText("0");
    positionZ->setText("0");

    rotationX->setText("0");
    rotationY->setText("0");
    rotationZ->setText("0");

    scaleX->setText("1");
    scaleY->setText("1");
    scaleZ->setText("1");


    openFilePath0->setText(NULL);
    openFilePath1->setText(NULL);
    openFilePath2->setText(NULL);

}

//**************************************************************************************
//When the Ok Button is clicked, this method is called: we just have to emit a signal to the parent, with the information on the object
void AddPreset::accept()
{
    std::string position;
    std::string rotation;
    std::string scale;

    std::string filenames[3];
    //In case of static objects
    if (openFileText0->isVisible())
        filenames[0] = openFilePath0->text().ascii();
    else
        filenames[0]=openFilePath2->text().ascii();

    filenames[1] = openFilePath1->text().ascii();
    filenames[2] = openFilePath2->text().ascii();

    std::ostringstream out;
    out << positionX->text().ascii()<<" "<<positionY->text().ascii()<<" "<<positionZ->text().ascii();
    position=out.str();
    out.str("");
    out << rotationX->text().ascii()<<" "<<rotationY->text().ascii()<<" "<<rotationZ->text().ascii();
    rotation=out.str();
    out.str("");
    out << scaleX->text().ascii()<<" "<<scaleY->text().ascii()<<" "<<scaleZ->text().ascii();
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
    fileName=std::string (s.ascii());

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

