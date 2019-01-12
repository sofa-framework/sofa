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
#ifndef QCOMPLIANTMOUSEOPERATIONS_H
#define QCOMPLIANTMOUSEOPERATIONS_H

#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/qt/SofaMouseManager.h>

#ifdef SOFA_QT4
#include <QWidget>
#include <QLineEdit>
#include <QRadioButton>
#include <QSpinBox>
#include <QSlider>
#include <QPushButton>
#include <QCheckBox>
#include <QGroupBox>
#include <QComboBox>
#else
#include <qwidget.h>
#include <qlineedit.h>
#include <qradiobutton.h>
#include <qspinbox.h>
#include <qslider.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#endif
#include <iostream>

#include <sofa/gui/qt/QMouseOperations.h>
#include "CompliantAttachPerformer.h"

namespace sofa
{

namespace gui
{

namespace qt
{
class DataWidget;



class SOFA_Compliant_gui_API QCompliantAttachOperation : public QMouseOperation, public CompliantAttachOperation
{
    Q_OBJECT
public:
    QCompliantAttachOperation();
    void configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button);

protected:
    DataWidget *complianceWidget;
    DataWidget *isComplianceWidget;
    DataWidget *arrowSizeWidget;
    DataWidget *colorWidget;
    DataWidget *visualModelWidget;
};


}
}
}

#endif
