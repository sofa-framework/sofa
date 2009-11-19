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

#include <sofa/gui/qt/QMouseOperations.h>
#ifdef SOFA_QT4
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
/*#include <QRadioButton>
#include <QPushButton>*/
#else
#include <qlayout.h>
#include <qlabel.h>
#include <qgroupbox.h>
/*#include <qradiobutton.h>
#include <qpushbutton.h>*/
#endif

namespace sofa
{

namespace gui
{

namespace qt
{
QAttachOperation::QAttachOperation()
{
    //Building the GUI for the Attach Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label=new QLabel(QString("Stiffness"), this);
    value=new QLineEdit(QString("1000.0"), this);

    layout->addWidget(label);
    layout->addWidget(value);
}

double QAttachOperation::getStiffness() const
{
    return atof(value->displayText().ascii());
}


QInciseOperation::QInciseOperation()
{
    //Building the GUI for the Injection Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    incisionMethodChoiceGroup = new QGroupBox(tr("Incision method choice"),this);

    method1 = new QRadioButton(tr("&Throw segment: Incise from click to click."), incisionMethodChoiceGroup);
    method2 = new QRadioButton(tr("&Continually: Incise continually from first click localization."), incisionMethodChoiceGroup);

    method1->setChecked (true);

    QVBoxLayout *vbox = new QVBoxLayout;
    vbox->addWidget(method1);
    vbox->addWidget(method2);
    //	vbox->addStretch(1);
    incisionMethodChoiceGroup->setLayout(vbox);
    layout->addWidget(incisionMethodChoiceGroup);
}

int QInciseOperation::getIncisionMethod() const
{
    if (method2->isChecked())
        return 1;
    else
        return 0;
}


QFixOperation::QFixOperation()
{
    //Building the GUI for the Fix Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label=new QLabel(QString("Fixation"), this);
    value=new QLineEdit(QString("10000.0"), this);

    layout->addWidget(label);
    layout->addWidget(value);
}

double QFixOperation::getStiffness() const
{
    return atof(value->displayText().ascii());
}

QInjectOperation::QInjectOperation()
{
    //Building the GUI for the Injection Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label1=new QLabel(QString("Potential Value"), this);
    value=new QLineEdit(QString("100.0"), this);

    QLabel *label2=new QLabel(QString("State Tag"), this);
    tag=new QLineEdit(QString("elec"), this);

    layout->addWidget(label1);
    layout->addWidget(value);

    layout->addWidget(label2);
    layout->addWidget(tag);
}

double QInjectOperation::getPotentialValue() const
{
    return atof(value->displayText().ascii());
}

std::string QInjectOperation::getStateTag() const
{
    return (std::string)(tag->displayText()).ascii();
}

}
}
}
