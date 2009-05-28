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

#include "ComponentLibrary.h"

#ifdef SOFA_QT4
#include <QToolTip>
#else
#include <qtooltip.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

//-------------------------------------------------------------------------------------------------------
ComponentLibrary::ComponentLibrary(QWidget *parent, const std::string &componentN, const std::string &categoryN, ClassEntry *e, const std::vector< QString > &exampleFiles): QWidget(parent, componentN.c_str()), name(componentN), categoryName(categoryN),entry(e)
{

    description  = std::string("<H2>")  + entry->className + std::string(": ");

    std::vector< std::string > possiblePaths;
    for (std::set< std::string >::iterator it=entry->baseClasses.begin(); it!=entry->baseClasses.end() ; it++)
    {
        if (it != entry->baseClasses.begin()) description += std::string(", ");
        description += (*it);
    }

    //Find a scene
    for (unsigned int i=0; i<exampleFiles.size(); ++i)
    {
        if (exampleFiles[i].findRev(entry->className.c_str()) >= 0 )
            possiblePaths.push_back(exampleFiles[i].ascii());
    }

    std::string nameSpace = sofa::core::objectmodel::Base::decodeNamespaceName(entry->creatorList.begin()->second->type());


    description += std::string("</H2>");

    description += std::string("<ul>");

    description += std::string("<li><b>Description: </b>") + entry->description + std::string("</li>");


    if (!nameSpace.empty())
        description += std::string("<li><b>NameSpace: </b>")+nameSpace +std::string("</li>");
    if (!entry->authors.empty())
        description += std::string("<li><b>Authors: </b>")+entry->authors +std::string("</li>");
    if (!entry->license.empty())
        description += std::string("<li><b>License: </b>") + entry->license + std::string("</li>");

    if (possiblePaths.size() != 0)
    {
        description += std::string("<li><b>Example: </b><ul>");
        for (unsigned int i=0; i<possiblePaths.size(); ++i)
        {
            description += std::string("<li><a href=\"")+possiblePaths[i]+std::string("\">") + possiblePaths[i] + std::string("</a></li>");
        }
        description += std::string("</ul>");
    }

    description += std::string("</ul>");
}





void ComponentLibrary::addTemplate( const std::string &nameT)
{
    if (nameT.empty()) return;
    templateName.push_back(nameT);
}


void ComponentLibrary::endConstruction()
{
}

//-------------------------------------------------------------------------------------------------------
QComponentLibrary::QComponentLibrary(QWidget *parent, ComponentLayout *l, const std::string &componentN, const std::string &categoryN, ClassEntry *e, const std::vector< QString > &exampleFiles): ComponentLibrary(parent, componentN,categoryN,e, exampleFiles)
{
    //-----------------------------------------------------------------------
    //QT Creation
    //-----------------------------------------------------------------------
//         layout    = new ComponentLayout( this );
    layout    = l;
    label     = new ComponentLabel( QString(name.c_str()), parent);
    templates = new ComponentTemplates(parent);

    connect( label, SIGNAL(pressed()), this, SLOT( componentPressed() ));

    const unsigned int row=layout->numRows();

    label->setFlat(false);
    std::string tooltipText = entry->description.substr(0, entry->description.size()-1);
    QToolTip::add(label, tooltipText.c_str());
    layout->addWidget(label,row,0);
    layout->addWidget(templates,row,1);
    templates->setHidden(true);
}

QComponentLibrary::~QComponentLibrary()
{
    //Shared layout
//         delete layout;
    delete label;
    delete templates;
}

void QComponentLibrary::endConstruction()
{
    if (templateName.empty()) return;

    templates->setHidden(false);
    for (unsigned int i=0; i<templateName.size(); ++i)
    {
        templates->insertItem(QString(templateName[i].c_str()));
    }
}

void QComponentLibrary::setDisplayed(bool b)
{
    if (b)
    {
//             this->show();
        label->show();
        if (!templateName.empty()) templates->show();
    }
    else
    {
//             this->hide();
        label->hide();
        if (!templateName.empty()) templates->hide();
    }
}

//*********************//
// SLOTS               //
//*********************//
void QComponentLibrary::componentPressed()
{
    std::string tName;
    if (!templateName.empty()) tName = templates->currentText().ascii();

    emit( componentDragged( description, tName, entry));
    label->setDown(false);
}
}
}
}
