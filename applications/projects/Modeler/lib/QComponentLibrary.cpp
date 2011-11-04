/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "QComponentLibrary.h"

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
QComponentLibrary::QComponentLibrary(QWidget *parent, ComponentLayout *l, const std::string &componentN, const std::string &categoryN, ClassEntry* e, const std::vector< std::string > &exampleFiles): QWidget(parent, componentN.c_str()), ComponentLibrary(componentN,categoryN,e, exampleFiles)
{
//         layout    = new ComponentLayout( this );
    layout    = l;
    label     = new ComponentLabel( QString(this->getName().c_str()), parent);
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
