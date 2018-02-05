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
#include "RGBAColorDataWidget.h"

namespace sofa
{
namespace gui
{
namespace qt
{
namespace rgbacolordatawidget_h
{

helper::Creator<DataWidgetFactory, RGBAColorDataWidget> DWClass("default",true);

bool RGBAColorDataWidget::createWidgets()
{

    m_colorPicker = new QRGBAColorPicker(this);

    QHBoxLayout* hlayout = new QHBoxLayout(this);
    hlayout->addWidget(m_colorPicker);

    connect(m_colorPicker, SIGNAL( hasChanged() ), this, SLOT( setWidgetDirty() ) );

    readFromData();

    return true;
}

void RGBAColorDataWidget::setDataReadOnly(bool readOnly)
{
    m_colorPicker->setEnabled(!readOnly);
}

void RGBAColorDataWidget::readFromData()
{
    m_colorPicker->setColor( getData()->getValue() );
}

void RGBAColorDataWidget::writeToData()
{
    RGBAColor* color = getData()->virtualBeginEdit();
    (*color) = m_colorPicker->getColor() ;

    getData()->virtualEndEdit();
}

} /// namespace rgbacolordatawidget
} /// namespace qt
} /// namespace gui
} /// namespace sofa
