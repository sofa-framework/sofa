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

    layout->add(openFilePath);
    layout->add(openFileButton);
    connect( openFileButton, SIGNAL( clicked() ), this, SLOT( raiseFileDialog() ) );
    connect( openFilePath, SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    return true;
}


void DataFileNameWidget::readFromData()
{
    const std::string& filepath = this->getData()->virtualGetValue();
    if (openFilePath->text().ascii() != filepath)
        openFilePath->setText(QString(filepath.c_str()) );
}

void DataFileNameWidget::writeToData()
{
    std::string fileName( openFilePath->text().ascii() );
    if (this->getData()->getValueString() != fileName)
        this->getData()->virtualSetValue(fileName);

}


void DataFileNameWidget::raiseDialog()
{
    std::string fileName( openFilePath->text().ascii() );

    if (sofa::helper::system::DataRepository.findFile(fileName))
        fileName=sofa::helper::system::DataRepository.getFile(fileName);
    else
        fileName=sofa::helper::system::DataRepository.getFirstPath();

    QString s  = getOpenFileName(this, QString(fileName.c_str()), "All (*)", "open file dialog",  "Choose a file to open" );
    std::string SofaPath = sofa::helper::system::DataRepository.getFirstPath();


    if (s.isNull() ) return;
    fileName=std::string (s.ascii());
//
//#ifdef WIN32
//
//  /* WIN32 is a pain here because of mixed case formatting with randomly
//  picked slash and backslash to separate dirs
//  */
//  std::replace(fileName.begin(),fileName.end(),'\\' , '/' );
//  std::replace(SofaPath.begin(),SofaPath.end(),'\\' , '/' );
//  std::transform(fileName.begin(), fileName.end(), fileName.begin(), ::tolower );
//  std::transform(SofaPath.begin(), SofaPath.end(), SofaPath.begin(), ::tolower );
//
//#endif
//	std::string::size_type loc = fileName.find( SofaPath, 0 );
//	if (loc==0) fileName = fileName.substr(SofaPath.size()+1);
    fileName = sofa::helper::system::FileRepository::relativeToPath(fileName,SofaPath);


    openFilePath->setText( QString( fileName.c_str() ) );
}


}
}
}

