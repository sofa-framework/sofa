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

#include "ModifierCondition.h"


#include <QHBoxLayout>


namespace sofa
{

namespace gui
{

namespace qt
{


QNamingModifierCondition::QNamingModifierCondition(QWidget *parent):QWidget(parent)
{
    //Build GUI
    QHBoxLayout *layout = new QHBoxLayout(this);

    activated = new QCheckBox(QString("Data name is "),this);
    criteriaSelector = new QComboBox(this);
    criteriaList << QString("Equal") << QString("Different");
    criteriaSelector->addItems(criteriaList);
    entryName = new QLineEdit(this);


    layout->addWidget(activated);
    layout->addWidget(criteriaSelector);
    layout->addWidget(entryName);
}

bool QNamingModifierCondition::verify(core::objectmodel::Base* c, core::objectmodel::BaseData* ) const
{
    if (!isActive()) return true;

    const std::string componentName=c->getName();
    const std::string criteria=criteriaSelector->currentText().toStdString();
    if (criteria == std::string("Equal"))
        return (componentName == getValue());
    else
        return (componentName != getValue());
}



QValueModifierCondition::QValueModifierCondition(QWidget *parent):QWidget(parent)
{
    //Build GUI
    QHBoxLayout *layout = new QHBoxLayout(this);

    activated = new QCheckBox(QString("Value is "),this);
    criteriaSelector = new QComboBox(this);
    criteriaList << QString("!=") << QString(">") << QString(">=") << QString("<") << QString("<=");
    criteriaSelector->addItems(criteriaList);

    value = new QLineEdit(this);


    layout->addWidget(activated);
    layout->addWidget(criteriaSelector);
    layout->addWidget(value);
}

bool QValueModifierCondition::verify(core::objectmodel::Base* /*c*/, core::objectmodel::BaseData* d) const
{
    if (!isActive()) return true;

    const std::string dataValue=d->getValueString();
    const std::string criteria=criteriaSelector->currentText().toStdString();

    if (criteria == std::string("!="))
        return (dataValue != getValue());
    else if (criteria == std::string(">"))
        return (dataValue > getValue());
    else if (criteria == std::string(">="))
        return (dataValue >= getValue());
    else if (criteria == std::string("<"))
        return (dataValue < getValue());
    else if (criteria == std::string("<="))
        return (dataValue <= getValue());
    else
        return false;
}


}
}
}
