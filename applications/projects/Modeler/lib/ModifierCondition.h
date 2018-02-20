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
#ifndef SOFA_MODIFIERCONDITION_H
#define SOFA_MODIFIERCONDITION_H

#include <sofa/core/objectmodel/Base.h>

#include <QCheckBox>
#include <QLineEdit>
#include <QComboBox>
#include <QStringList>

namespace sofa
{

namespace gui
{

namespace qt
{


struct ModifierCondition
{
    virtual ~ModifierCondition() {};
    virtual bool verify(core::objectmodel::Base* c, core::objectmodel::BaseData* d) const =0;
    virtual bool isActive() const=0;
};




class QNamingModifierCondition: public QWidget, public ModifierCondition
{
    Q_OBJECT
public:
    QNamingModifierCondition(QWidget *parent=0);

    bool verify(core::objectmodel::Base* c, core::objectmodel::BaseData* d) const;
    bool isActive() const {return activated->isChecked();}
    std::string getValue() const {return entryName->text().toStdString();}
protected:
    QCheckBox *activated;
    QLineEdit *entryName;
    QComboBox *criteriaSelector;
    QStringList criteriaList;
};


class QValueModifierCondition: public QWidget, public ModifierCondition
{
    Q_OBJECT
public:
    QValueModifierCondition(QWidget *parent=0);

    bool verify(core::objectmodel::Base* c, core::objectmodel::BaseData* d) const;

    bool isActive() const {return activated->isChecked();}
    std::string getValue() const {return value->text().toStdString();}
protected:
    QCheckBox *activated;
    QLineEdit *value;
    QComboBox *criteriaSelector;
    QStringList criteriaList;
};



}
}
}

#endif
