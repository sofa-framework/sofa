/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <PluginExample/MyDataWidgetUnsigned.h>

#include <sofa/helper/Factory.h>


namespace sofa::gui::qt
{

/*
 * Register this new class in the DataWidgetFactory.  The factory key is the
 * Data widget property (see MyBehaviorModel constructor).
 */
helper::Creator<DataWidgetFactory, MyDataWidgetUnsigned> DW_myData("widget_myData",false);

bool MyDataWidgetUnsigned::createWidgets()
{
    unsigned myData_value = getData()->virtualGetValue();

    m_qslider = new QSlider(Qt::Horizontal, this);
    m_qslider->setTickPosition(QSlider::TicksBelow);
    m_qslider->setRange(0, 100);
    m_qslider->setValue((int)myData_value);

    QString label1_text("Data current value = ");
    label1_text.append(getData()->getValueString().c_str());
    m_label1 = new QLabel(this);
    m_label1->setText(label1_text);

    QString label2_text = "Data value after updating = ";
    label2_text.append(QString().setNum(m_qslider->value()));
    m_label2 = new QLabel(this);
    m_label2->setText(label2_text);


    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(m_label1);
    layout->addWidget(m_label2);
    layout->addWidget(m_qslider);

    connect(m_qslider, SIGNAL(sliderReleased()), this, SLOT(setWidgetDirty()));
    connect(m_qslider, SIGNAL(valueChanged(int)), this, SLOT(setWidgetDirty()));
    connect(m_qslider, SIGNAL(sliderReleased()), this, SLOT(change()));
    connect(m_qslider, SIGNAL(valueChanged(int)), this, SLOT(change()));

    return true;
}

void MyDataWidgetUnsigned::setDataReadOnly(bool readOnly)
{
    m_qslider->setEnabled(!readOnly);
}

void MyDataWidgetUnsigned::readFromData()
{
    m_qslider->setValue((int)getData()->virtualGetValue());

    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());

    QString label2_text = "myData value after updating = ";
    label2_text.append(QString().setNum(m_qslider->value()));

    m_label1->setText(label1_text);
    m_label2->setText(label2_text);

}

void MyDataWidgetUnsigned::writeToData()
{
    unsigned widget_value = (unsigned)m_qslider->value();
    getData()->virtualSetValue(widget_value);

    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());
    QString label2_text = "myData value after updating = ";
    label2_text.append(QString().setNum(m_qslider->value()));

    m_label1->setText(label1_text);
    m_label2->setText(label2_text);
}

void MyDataWidgetUnsigned::change()
{
    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());
    QString label2_text = "myData value after updating = ";
    label2_text.append(QString().setNum(m_qslider->value()));

    m_label1->setText(label1_text);
    m_label2->setText(label2_text);
}


} // namespace sofa::gui::qt
