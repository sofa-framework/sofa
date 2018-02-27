/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "DataFilenameWidget.h"
#include <sofa/helper/Factory.h>

#include "FileManagement.h" //static functions to manage opening/ saving of files
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>


#include <algorithm>

namespace sofa
{
namespace gui
{
namespace qt
{

helper::Creator<DataWidgetFactory,DataFileNameWidget> DW_Datafilename("widget_filename",false);



bool DataFileNameWidget::createWidgets()
{
    QHBoxLayout* layout = new QHBoxLayout(this);

    openFilePath = new QLineEdit(this);
    const std::string& filepath = this->getData()->virtualGetValue();
    openFilePath->setText( QString(filepath.c_str()) );

    openFileButton = new QPushButton(this);
    openFileButton->setText("...");

    layout->addWidget(openFilePath);
    layout->addWidget(openFileButton);
    connect( openFileButton, SIGNAL( clicked() ), this, SLOT( raiseDialog() ) );
    connect( openFilePath, SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    return true;
}

void DataFileNameWidget::setDataReadOnly(bool readOnly)
{
    openFilePath->setReadOnly(readOnly);
    openFileButton->setEnabled(!readOnly);
}

void DataFileNameWidget::readFromData()
{
    const std::string& filepath = this->getData()->getValue();
    if (openFilePath->text().toStdString() != filepath)
        openFilePath->setText(QString(filepath.c_str()) );
}

void DataFileNameWidget::writeToData()
{
    std::string fileName( openFilePath->text().toStdString() );
    if (this->getData()->getValueString() != fileName)
        this->getData()->setValue(fileName);

}


void DataFileNameWidget::raiseDialog()
{
    std::string fileName( openFilePath->text().toStdString() );

    if (sofa::helper::system::DataRepository.findFile(fileName))
        fileName=sofa::helper::system::DataRepository.getFile(fileName);
    else
        fileName=sofa::helper::system::DataRepository.getFirstPath();

    QString s  = getOpenFileName(this, QString(fileName.c_str()), "All (*)", "open file dialog",  "Choose a file to open" );
    std::string SofaPath = sofa::helper::system::DataRepository.getFirstPath();


    if (s.isNull() ) return;
    fileName=std::string (s.toStdString());
    fileName = sofa::helper::system::FileRepository::relativeToPath(fileName,SofaPath,false);

    openFilePath->setText( QString( fileName.c_str() ) );
}


}
}
}

