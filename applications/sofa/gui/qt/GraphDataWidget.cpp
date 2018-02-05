/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include "GraphDataWidget.h"
#include "DataWidget.h"
#include <sofa/helper/Factory.inl>
#include <sofa/helper/map.h>
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Creator;
using sofa::helper::fixed_array;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(GraphDataWidget);


Creator<DataWidgetFactory, GraphDataWidget< std::map< std::string, sofa::helper::vector<float> > > > DWClass_mapvectorf("graph",true);
Creator<DataWidgetFactory, GraphDataWidget< std::map< std::string, sofa::helper::vector<double> > > > DWClass_mapvectord("graph",true);
Creator<DataWidgetFactory, GraphDataWidget< std::map< std::string, sofa::helper::vector<Vec2d> > > > DWClass_mapvector2d("graph",true);
Creator<DataWidgetFactory, GraphDataWidget< std::map< std::string, std::deque<float> > > > DWClass_mapdequef("graph",true);
Creator<DataWidgetFactory, GraphDataWidget< std::map< std::string, std::deque<double> > > > DWClass_mapdequed("graph",true);
Creator<DataWidgetFactory, GraphDataWidget< std::map< std::string, std::deque<Vec2d> > > > DWClass_mapdeque2d("graph",true);

Creator<DataWidgetFactory, GraphDataWidget_Linear< std::map< std::string, sofa::helper::vector<float> > > > DWLClass_mapvectorf("graph_linear",true);
Creator<DataWidgetFactory, GraphDataWidget_Linear< std::map< std::string, sofa::helper::vector<double> > > > DWLClass_mapvectord("graph_linear",true);
Creator<DataWidgetFactory, GraphDataWidget_Linear< std::map< std::string, sofa::helper::vector<Vec2d> > > > DWLClass_mapvector2d("graph_linear",true);
Creator<DataWidgetFactory, GraphDataWidget_Linear< std::map< std::string, std::deque<float> > > > DWLClass_mapdequef("graph_linear",true);
Creator<DataWidgetFactory, GraphDataWidget_Linear< std::map< std::string, std::deque<double> > > > DWLClass_mapdequed("graph_linear",true);
Creator<DataWidgetFactory, GraphDataWidget_Linear< std::map< std::string, std::deque<Vec2d> > > > DWLClass_mapdeque2d("graph_linear",true);



GraphOptionWidget::GraphOptionWidget(const std::string &dataName, GraphSetting *graphConf):graph(graphConf)
{
    QVBoxLayout *generalLayout = new QVBoxLayout(this);
    generalLayout->setMargin(0);
    generalLayout->setSpacing(0);
    QWidget *gnuplotExport=new QWidget(this);
    QHBoxLayout* gnuplotLayout = new QHBoxLayout(gnuplotExport);
    exportGNUPLOTButton = new QPushButton(QString("GNUPLOT"), gnuplotExport);

    //Create field to enter the base name of the gnuplot files
    fileGNUPLOTLineEdit = new QLineEdit(gnuplotExport);
    findGNUPLOTFile = new QPushButton(QString("..."), gnuplotExport);

    std::string gnuplotDirectory; //=simulation::getSimulation()->gnuplotDirectory.getValue();
    if (gnuplotDirectory.empty())
        gnuplotDirectory=sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + "/";
    gnuplotDirectory += dataName;
    fileGNUPLOTLineEdit->setText(QString(gnuplotDirectory.c_str()));

    gnuplotLayout->addWidget(exportGNUPLOTButton);
    gnuplotLayout->addWidget(fileGNUPLOTLineEdit);
    gnuplotLayout->addWidget(findGNUPLOTFile);
    generalLayout->addWidget(gnuplotExport);

    connect(exportGNUPLOTButton, SIGNAL(clicked()), this, SLOT(exportGNUPlot()));
    connect(findGNUPLOTFile, SIGNAL(clicked()), this, SLOT(openFindFileDialog()));

    QWidget *imageExport=new QWidget(this);
    QHBoxLayout* imageLayout = new QHBoxLayout(imageExport);

    checkBox = new QCheckBox("Check the box to dump the graph at each edit of the data");

    exportImageButton = new QPushButton(QString("Image"), imageExport);
    //Create field to enter the base name of the gnuplot files
    fileImageLineEdit = new QLineEdit(imageExport);
    findImageFile = new QPushButton(QString("..."), imageExport);

    std::string imageDirectory; //=simulation::getSimulation()->gnuplotDirectory.getValue();
    if (imageDirectory.empty())
        imageDirectory=sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + "/";
    imageDirectory += dataName;
    fileImageLineEdit->setText(QString(imageDirectory.c_str()));

    imageLayout->addWidget(exportImageButton);
    imageLayout->addWidget(fileImageLineEdit);
    imageLayout->addWidget(findImageFile);
    generalLayout->addWidget(imageExport);

    connect(exportImageButton, SIGNAL(clicked()), this, SLOT(exportImage()));
    connect(findImageFile, SIGNAL(clicked()), this, SLOT(openFindFileDialog()));

    generalLayout->addWidget(checkBox);

    idfile = 0;
}

void GraphOptionWidget::openFindFileDialog()
{
    QLineEdit *fileLineEdit=0;
    QPushButton *button=(QPushButton*)sender();
    if (button == findGNUPLOTFile)    fileLineEdit = fileGNUPLOTLineEdit;

    else if (button == findImageFile) fileLineEdit = fileImageLineEdit;

    std::string filename(sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()));
    std::string directory;
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        directory = s.toStdString();
        if (directory.at(directory.size()-1) != '/') directory+="/";
        fileLineEdit->setText(directory.c_str());
    }
}
void GraphOptionWidget::exportGNUPlot()
{
    graph->exportGNUPlot(fileGNUPLOTLineEdit->text().toStdString());
}

void GraphOptionWidget::exportImage()
{
    const unsigned nbpad=5;

    std::stringstream ss;
    ss << idfile;
    std::string idstring = ss.str();

    std::string pad(nbpad-idstring.length(),'0');
    idfile++;
    if (idfile>99999) idfile = 0;

    std::string filename = fileImageLineEdit->text().toStdString();
    filename.append("_");
    filename.append(pad);
    filename.append(idstring);
    graph->exportImage(filename);
}

} // namespace qt

} // namespace gui

} // namespace sofa
