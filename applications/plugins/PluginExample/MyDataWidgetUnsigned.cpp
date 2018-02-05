/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "MyDataWidgetUnsigned.h"

#include <sofa/helper/Factory.h>


namespace sofa
{

namespace gui
{

namespace qt
{


/*
 * Register this new class in the DataWidgetFactory.  The factory key is the
 * Data widget property (see MyBehaviorModel constructor).
 */
helper::Creator<DataWidgetFactory, MyDataWidgetUnsigned> DW_myData("widget_myData",false);

bool MyDataWidgetUnsigned::createWidgets()
{
    unsigned myData_value = getData()->virtualGetValue();

    qslider = new QSlider(Qt::Horizontal, this);
    qslider->setTickPosition(QSlider::TicksBelow);
    qslider->setRange(0, 100);
    qslider->setValue((int)myData_value);

    QString label1_text("Data current value = ");
    label1_text.append(getData()->getValueString().c_str());
    label1 = new QLabel(this);
    label1->setText(label1_text);

    QString label2_text = "Data value after updating = ";
    label2_text.append(QString().setNum(qslider->value()));
    label2 = new QLabel(this);
    label2->setText(label2_text);


    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(label1);
    layout->addWidget(label2);
    layout->addWidget(qslider);

    connect(qslider, SIGNAL(sliderReleased()), this, SLOT(setWidgetDirty()));
    connect(qslider, SIGNAL(valueChanged(int)), this, SLOT(setWidgetDirty()));
    connect(qslider, SIGNAL(sliderReleased()), this, SLOT(change()));
    connect(qslider, SIGNAL(valueChanged(int)), this, SLOT(change()));

    return true;
}

void MyDataWidgetUnsigned::setDataReadOnly(bool readOnly)
{
    qslider->setEnabled(!readOnly);
}

void MyDataWidgetUnsigned::readFromData()
{
    qslider->setValue((int)getData()->virtualGetValue());

    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());

    QString label2_text = "myData value after updating = ";
    label2_text.append(QString().setNum(qslider->value()));

    label1->setText(label1_text);
    label2->setText(label2_text);

}

void MyDataWidgetUnsigned::writeToData()
{
    unsigned widget_value = (unsigned)qslider->value();
    getData()->virtualSetValue(widget_value);

    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());
    QString label2_text = "myData value after updating = ";
    label2_text.append(QString().setNum(qslider->value()));

    label1->setText(label1_text);
    label2->setText(label2_text);
}

void MyDataWidgetUnsigned::change()
{
    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());
    QString label2_text = "myData value after updating = ";
    label2_text.append(QString().setNum(qslider->value()));

    label1->setText(label1_text);
    label2->setText(label2_text);
}


} // namespace qt

} // namespace gui

} // namespace sofa
