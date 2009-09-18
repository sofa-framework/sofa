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
#include <sofa/gui/qt/SofaMouseManager.h>
#include <sofa/gui/qt/QMouseOperations.h>

#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/OperationFactory.h>

#include <iostream>
#ifndef SOFA_QT4
#include <qlineedit.h>
#include <qcombobox.h>
#include <qlabel.h>
#endif

namespace sofa
{
namespace gui
{
namespace qt
{
SofaMouseManager::SofaMouseManager()
{
    connect( LeftOperationCombo,   SIGNAL(activated(int)), this, SLOT( selectOperation(int) ));
    connect( MiddleOperationCombo, SIGNAL(activated(int)), this, SLOT( selectOperation(int) ));
    connect( RightOperationCombo,  SIGNAL(activated(int)), this, SLOT( selectOperation(int) ));

    LeftValue  ->setText(QString::number(1000.0));
    MiddleValue->setText(QString::number(1000.0));
    RightValue ->setText(QString::number(1000.0));

    RegisterOperation("Attach").add< QAttachOperation >();
    RegisterOperation("Fix")   .add< QFixOperation  >();
    RegisterOperation("Incise").add< InciseOperation  >();
    RegisterOperation("Remove").add< RemoveOperation  >();
    RegisterOperation("Sculpt").add< QSculptOperation  >();
}

void SofaMouseManager::setPickHandler(PickHandler *picker)
{
    pickHandler=picker;

    LeftOperationCombo->clear();
    MiddleOperationCombo->clear();
    RightOperationCombo->clear();
    mapIndexOperation.clear();

    const OperationFactory::RegisterStorage &registry = OperationFactory::getInstance()->registry;

    int idx=0;
    for (OperationFactory::RegisterStorage::const_iterator it=registry.begin(); it!=registry.end(); ++it)
    {
        LeftOperationCombo  ->insertItem(QString(OperationFactory::GetDescription(it->first).c_str()));
        MiddleOperationCombo->insertItem(QString(OperationFactory::GetDescription(it->first).c_str()));
        RightOperationCombo ->insertItem(QString(OperationFactory::GetDescription(it->first).c_str()));

        if      (it->first == "Attach") LeftOperationCombo->setCurrentItem(idx);
        else if (it->first == "Incise") MiddleOperationCombo->setCurrentItem(idx);
        else if (it->first == "Remove") RightOperationCombo->setCurrentItem(idx);
        mapIndexOperation.insert(std::make_pair(idx++, it->first));
    }



    updateOperation(LEFT,   "Attach");
    updateOperation(MIDDLE, "Incise");
    updateOperation(RIGHT,  "Remove");
}


void SofaMouseManager::selectOperation(int operation)
{
    QComboBox *combo = (QComboBox*)(sender());
    const std::string operationName=mapIndexOperation[operation];

    if      (combo == LeftOperationCombo)   updateOperation(LEFT,   operationName);
    else if (combo == MiddleOperationCombo) updateOperation(MIDDLE, operationName);
    else if (combo == RightOperationCombo)  updateOperation(RIGHT,  operationName);
}

void SofaMouseManager::setValue( MOUSE_BUTTON button, const char *text, double value )
{
    switch(button)
    {
    case LEFT:
        if (strcmp(text, "Force") == 0)
        {
            LeftValueLabelForce->setText(text);
            LeftSliderForce->setValue((int)value);
            return;
        }
        if (strcmp(text, "Scale") == 0)
        {
            LeftValueLabelScale->setText(text);
            LeftSliderScale->setValue((int)value);
            return;
        }
        else
        {
            LeftValueLabel->setText(text);
            LeftValue->setText(QString::number(value));
            return;
        }
    case MIDDLE:
        if (strcmp(text, "Force") == 0)
        {
            MiddleValueLabelForce->setText(text);
            MiddleSliderForce->setValue((int)value);
            return;
        }
        if (strcmp(text, "Scale") == 0)
        {
            MiddleValueLabelScale->setText(text);
            MiddleSliderScale->setValue((int)value);
            return;
        }
        else
        {
            MiddleValueLabel->setText(text);
            MiddleValue->setText(QString::number(value));
            return;
        }
    case RIGHT:
        if (strcmp(text, "Force") == 0)
        {
            RightValueLabelForce->setText(text);
            RightSliderForce->setValue((int)value);
            return;
        }
        if (strcmp(text, "Scale") == 0)
        {
            RightValueLabelScale->setText(text);
            RightSliderScale->setValue((int)value);
            return;
        }
        else
        {
            RightValueLabel->setText(text);
            RightValue->setText(QString::number(value));
            return;
        }
    }
}

double SofaMouseManager::getValue( MOUSE_BUTTON button, const char *text ) const
{
    switch(button)
    {
    case LEFT:
        if (strcmp(text, "Force")==0)
            return (double) LeftSliderForce->value()/5000;
        if (strcmp(text, "Scale")==0)
            return (double) (100 - LeftSliderScale->value())/5;
        else
            return atof(LeftValue->text().ascii());
    case MIDDLE:
        if (strcmp(text, "Force")==0)
            return (double) MiddleSliderForce->value()/5000;
        if (strcmp(text, "Scale")==0)
            return (double) (100 - MiddleSliderScale->value())/5;
        else
            return atof(MiddleValue->text().ascii());
    case RIGHT:
        if (strcmp(text, "Force")==0)
            return (double) RightSliderForce->value()/5000;
        if (strcmp(text, "Scale")==0)
            return (double) (100 - RightSliderScale->value())/5;
        else
            return atof(RightValue->text().ascii());
    }
    return 0;
}

void SofaMouseManager::updateOperation( MOUSE_BUTTON button, const std::string &id)
{
    switch(button)
    {
    case LEFT:
        if (OperationFactory::IsModifiable(id))
        {
            LeftValue->show(); LeftValueLabel->show();
            LeftValueLabelForce->hide(); LeftSliderForce->hide(); LeftSpinBoxForce->hide();
            LeftValueLabelScale->hide(); LeftSliderScale->hide(); LeftSpinBoxScale->hide();
        }
        else
        {
            LeftValue->hide(); LeftValueLabel->hide();
            if (id == "Sculpt")
            {
                LeftValueLabelForce->show(); LeftSliderForce->show(); LeftSpinBoxForce->show();
                LeftValueLabelScale->show(); LeftSliderScale->show(); LeftSpinBoxScale->show();
            }
            else
            {
                LeftValueLabelForce->hide(); LeftSliderForce->hide(); LeftSpinBoxForce->hide();
                LeftValueLabelScale->hide(); LeftSliderScale->hide(); LeftSpinBoxScale->hide();
            }
        }
        break;

    case MIDDLE:
        if (OperationFactory::IsModifiable(id))
        {
            MiddleValue->show(); MiddleValueLabel->show();
            MiddleValueLabelForce->hide(); MiddleSliderForce->hide(); MiddleSpinBoxForce->hide();
            MiddleValueLabelScale->hide(); MiddleSliderScale->hide(); MiddleSpinBoxScale->hide();
        }
        else
        {
            MiddleValue->hide(); MiddleValueLabel->hide();
            if (id == "Sculpt")
            {
                MiddleValueLabelForce->show(); MiddleSliderForce->show(); MiddleSpinBoxForce->show();
                MiddleValueLabelScale->show(); MiddleSliderScale->show(); MiddleSpinBoxScale->show();
            }
            else
            {
                MiddleValueLabelForce->hide(); MiddleSliderForce->hide(); MiddleSpinBoxForce->hide();
                MiddleValueLabelScale->hide(); MiddleSliderScale->hide(); MiddleSpinBoxScale->hide();
            }
        }
        break;

    case RIGHT:
        if (OperationFactory::IsModifiable(id))
        {
            RightValue->show(); RightValueLabel->show();
            RightValueLabelForce->hide(); RightSliderForce->hide(); RightSpinBoxForce->hide();
            RightValueLabelScale->hide(); RightSliderScale->hide(); RightSpinBoxScale->hide();
        }
        else
        {
            RightValue->hide(); RightValueLabel->hide();
            if (id == "Sculpt")
            {
                RightValueLabelForce->show(); RightSliderForce->show(); RightSpinBoxForce->show();
                RightValueLabelScale->show(); RightSliderScale->show(); RightSpinBoxScale->show();
            }
            else
            {
                RightValueLabelForce->hide(); RightSliderForce->hide(); RightSpinBoxForce->hide();
                RightValueLabelScale->hide(); RightSliderScale->hide(); RightSpinBoxScale->hide();
            }
        }
        break;
    }
    pickHandler->changeOperation( button, id);
}

}
}
}

