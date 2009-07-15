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
    RegisterOperation("Incise").add< InciseOperation  >();
    RegisterOperation("Remove").add< RemoveOperation  >();
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

        mapIndexOperation.insert(std::make_pair(idx++, it->first));
    }



    LeftOperationCombo->setCurrentItem(0);
    updateOperation(LEFT,   "Attach");
    MiddleOperationCombo->setCurrentItem(1);
    updateOperation(MIDDLE, "Incise");
    RightOperationCombo->setCurrentItem(2);
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

double SofaMouseManager::getValue( MOUSE_BUTTON button )
{
    switch(button)
    {
    case LEFT:   return atof(LeftValue->text().ascii());
    case MIDDLE: return atof(MiddleValue->text().ascii());
    case RIGHT:  return atof(RightValue->text().ascii());
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
        }
        else
        {
            LeftValue->hide(); LeftValueLabel->hide();
        }
        break;

    case MIDDLE:
        if (OperationFactory::IsModifiable(id))
        {
            MiddleValue->show(); MiddleValueLabel->show();
        }
        else
        {
            MiddleValue->hide(); MiddleValueLabel->hide();
        }
        break;

    case RIGHT:
        if (OperationFactory::IsModifiable(id))
        {
            RightValue->show(); RightValueLabel->show();
        }
        else
        {
            RightValue->hide(); RightValueLabel->hide();
        }
        break;
    }
    pickHandler->changeOperation( button, id);
}

}
}
}

