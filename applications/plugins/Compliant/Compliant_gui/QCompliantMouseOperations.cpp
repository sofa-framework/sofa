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

#include "QCompliantMouseOperations.h"
#include <sofa/gui/qt/QDisplayDataWidget.h>
#include <sofa/gui/qt/DataWidget.h>
#include "../misc/CompliantAttachButtonSetting.h"

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
#include <qcombobox.h>
/*#include <qradiobutton.h>
#include <qpushbutton.h>*/
#endif

namespace sofa
{

namespace gui
{

namespace qt
{


QCompliantAttachOperation::QCompliantAttachOperation()
{
    //Building the GUI for the CompliantAttach Operation

    QVBoxLayout *layout=new QVBoxLayout(this);
    QHBoxLayout *layout1=new QHBoxLayout();layout->addLayout(layout1);
    QHBoxLayout *layout2=new QHBoxLayout();layout->addLayout(layout2);


    QLabel *label=new QLabel(QString("Compliance"), this);
    complianceWidget = createWidgetFromData(&(this->setting->compliance));

    QLabel *labelIs=new QLabel(QString("IsCompliance"), this);
    isComplianceWidget = createWidgetFromData(&(this->setting->isCompliance));

    QLabel *labelSize=new QLabel(QString("Arrow Size"), this);
    arrowSizeWidget = createWidgetFromData(&(this->setting->arrowSize));

    QLabel *labelColor=new QLabel(QString("Color"), this);
    colorWidget = createWidgetFromData(&(this->setting->color));

    QLabel *labelVM=new QLabel(QString("Add VisualModel?"), this);
    visualModelWidget = createWidgetFromData(&(this->setting->visualmodel));


    layout1->addWidget(label);
    layout1->addWidget(complianceWidget);

    layout1->addWidget(labelIs);
    layout1->addWidget(isComplianceWidget);

    layout1->addWidget(labelVM);
    layout1->addWidget(visualModelWidget);

    layout2->addWidget(labelColor);
    layout2->addWidget(colorWidget);

    layout2->addWidget(labelSize);
    layout2->addWidget(arrowSizeWidget);

}


void QCompliantAttachOperation::configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button)
{
    if (sofa::component::configurationsetting::CompliantAttachButtonSetting* attachSetting=dynamic_cast<sofa::component::configurationsetting::CompliantAttachButtonSetting*>(button))
    {
        CompliantAttachOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
        setting->compliance.copyValue(&(attachSetting->compliance));
        setting->arrowSize.copyValue(&(attachSetting->arrowSize) );
        setting->isCompliance.copyValue(&( attachSetting->isCompliance) ) ;
        setting->color.copyValue(&( attachSetting->color) ) ;
        setting->visualmodel.copyValue(&( attachSetting->visualmodel) ) ;

        complianceWidget->updateWidgetValue();
        isComplianceWidget->updateWidgetValue();
        arrowSizeWidget->updateWidgetValue();
        visualModelWidget->updateWidgetValue();
    }
    else CompliantAttachOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
}

} // namespace sofa
} // namespace gui
} // namespace qt


