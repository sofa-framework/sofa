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

#include "QMouseOperations.h"
#include "QDisplayDataWidget.h"
#include "DataWidget.h"
#include <SofaGraphComponent/AttachBodyButtonSetting.h>
#include <SofaGraphComponent/AddRecordedCameraButtonSetting.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>


namespace sofa
{

namespace gui
{

namespace qt
{

DataWidget *QMouseOperation::createWidgetFromData(sofa::core::objectmodel::BaseData* data)
{
    DataWidget::CreatorArgument arg;
    arg.data = data;
    arg.name =  arg.data->getName();
    arg.parent = this;
    arg.readOnly = arg.data->isReadOnly();
    DataWidget *widget = DataWidget::CreateDataWidget(arg);
    connect(widget, SIGNAL(WidgetDirty(bool)), this, SLOT(WidgetDirty(bool)));
    return widget;
}

void QMouseOperation::WidgetDirty(bool b)
{
    if (b)
    {
        DataWidget *dataW=(DataWidget*) sender();

        dataW->updateDataValue();
    }
}


//*******************************************************************************************
QAttachOperation::QAttachOperation()
{
    //Building the GUI for the Attach Operation

    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label=new QLabel(QString("Stiffness"), this);
    stiffnessWidget = createWidgetFromData(&(setting->stiffness));

    QLabel *labelSize=new QLabel(QString("Arrow Size"), this);
    arrowSizeWidget = createWidgetFromData(&(setting->arrowSize));

    QLabel *labelShowFactor=new QLabel(QString("Show Factor Size"), this);
    showSizeFactorWidget = createWidgetFromData(&(setting->showFactorSize));

    layout->addWidget(label);
    layout->addWidget(stiffnessWidget);

    layout->addWidget(labelSize);
    layout->addWidget(arrowSizeWidget);

    layout->addWidget(labelShowFactor);
    layout->addWidget(showSizeFactorWidget);
}


void QAttachOperation::configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button)
{
    if (sofa::component::configurationsetting::AttachBodyButtonSetting* attachSetting=dynamic_cast<sofa::component::configurationsetting::AttachBodyButtonSetting*>(button))
    {
        AttachOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
        setting->stiffness.copyValue(&(attachSetting->stiffness));
        setting->arrowSize.copyValue(&(attachSetting->arrowSize) );
        setting->showFactorSize.copyValue(&( attachSetting->showFactorSize) ) ;

        stiffnessWidget->updateWidgetValue();
        arrowSizeWidget->updateWidgetValue();
    }
    else AttachOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
}
//*******************************************************************************************


//*******************************************************************************************
QAddRecordedCameraOperation::QAddRecordedCameraOperation()
{}

void QAddRecordedCameraOperation::configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button)
{
    if (/*sofa::component::configurationsetting::AddRecordedCameraButtonSetting* cameraSetting=*/dynamic_cast<sofa::component::configurationsetting::AddRecordedCameraButtonSetting*>(button))
    {
        AddRecordedCameraOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
    }
    else AddRecordedCameraOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
}

//*******************************************************************************************


//*******************************************************************************************
QStartNavigationOperation::QStartNavigationOperation()
{}

void QStartNavigationOperation::configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button)
{
    if (/*sofa::component::configurationsetting::StartNavigationButtonSetting* cameraSetting=*/dynamic_cast<sofa::component::configurationsetting::StartNavigationButtonSetting*>(button))
    {
        StartNavigationOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
    }
    else StartNavigationOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
}

//*******************************************************************************************


//*******************************************************************************************
QFixOperation::QFixOperation()
{
    //Building the GUI for the Fix Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label=new QLabel(QString("Fixation"), this);
    stiffnessWidget = createWidgetFromData(&setting->stiffness);

    layout->addWidget(label);
    layout->addWidget(stiffnessWidget);
}

void QFixOperation::configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button)
{
    if (sofa::component::configurationsetting::FixPickedParticleButtonSetting* fixSetting=dynamic_cast<sofa::component::configurationsetting::FixPickedParticleButtonSetting*>(button))
    {
        FixOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId() )) ;
        setting->stiffness.setValue(fixSetting->stiffness.getValue());

        stiffnessWidget->updateWidgetValue();
    }
    else FixOperation::configure(picker,GetMouseId(button->button.getValue().getSelectedId()));
}

//*******************************************************************************************

//*******************************************************************************************
QInciseOperation::QInciseOperation()
    : finishIncision(0)
    , keepPoint(0)
{
    //Building the GUI for the Injection Operation
    QVBoxLayout *layout=new QVBoxLayout(this);

    // First group box for incision method choice
    incisionMethodChoiceGroup = new QGroupBox(tr("Incision method choice"),this);
    QVBoxLayout *vbox1 = new QVBoxLayout(incisionMethodChoiceGroup);

    method1 = new QRadioButton(tr("&Through segment: Incise from click to click."), incisionMethodChoiceGroup);
    method2 = new QRadioButton(tr("&Continually: Incise continually from first click localization."), incisionMethodChoiceGroup);
    method1->setChecked (true);

    vbox1->addWidget(method1);
    vbox1->addWidget(method2);

    // Second group box for easy use
    advancedOperations = new QGroupBox(tr("Advanced operations"),this);
    QVBoxLayout *vbox2 = new QVBoxLayout(advancedOperations);

    // on push button and one check box with labels
    finishCut = new QCheckBox(QString("Complete incision"),advancedOperations);
    storeLastPoint = new QCheckBox (QString("Keep in memory last incision point."),advancedOperations);

    vbox2->addWidget(finishCut);
    vbox2->addWidget(storeLastPoint);


    // Third group box for advanced settings (only snping % value for the moment)
    advancedOptions = new QGroupBox(tr("Advanced settings"),this);
    QVBoxLayout *vbox3 = new QVBoxLayout(advancedOptions);

    // first slider for border snaping
    QHBoxLayout *slider1=new QHBoxLayout();
    QLabel *label1=new QLabel(QString("Distance to snap from border (in %)"), this);
    snapingBorderSlider=new QSlider(Qt::Horizontal, this);
    snapingBorderValue=new QSpinBox(this);
    snapingBorderValue->setMinimum(0);
    snapingBorderValue->setMaximum(100);
    snapingBorderValue->setSingleStep(1);
    snapingBorderValue->setEnabled(true);

    slider1->addWidget (label1);
    slider1->addWidget (snapingBorderSlider);
    slider1->addWidget (snapingBorderValue);
    vbox3->addLayout (slider1);

    // second slider for along path snaping
    QHBoxLayout *slider2=new QHBoxLayout();
    QLabel *label2=new QLabel(QString("Distance to snap along path (in %)"), this);
    snapingSlider=new QSlider(Qt::Horizontal, this);
    snapingValue=new QSpinBox(this);
    snapingBorderValue->setMinimum(0);
    snapingBorderValue->setMaximum(100);
    snapingBorderValue->setSingleStep(1);
    snapingValue->setEnabled(true);
    snapingBorderValue->setValue(0);

    slider2->addWidget (label2);
    slider2->addWidget (snapingSlider);
    slider2->addWidget (snapingValue);
    vbox3->addLayout (slider2);

    // Creating UI
    layout->addWidget(incisionMethodChoiceGroup);
    layout->addWidget(advancedOperations);
    layout->addWidget(advancedOptions);

    connect(finishCut, SIGNAL(toggled(bool)), this, SLOT(setFinishIncision(bool)));
    connect(storeLastPoint, SIGNAL(toggled(bool)), this, SLOT(setkeepPoint(bool)));

    connect(snapingBorderSlider,SIGNAL(valueChanged(int)), snapingBorderValue, SLOT(setValue(int)));
    connect(snapingBorderValue,SIGNAL(valueChanged(int)), snapingBorderSlider, SLOT(setValue(int)));

    connect(snapingSlider,SIGNAL(valueChanged(int)), snapingValue, SLOT(setValue(int)));
    connect(snapingValue,SIGNAL(valueChanged(int)), snapingSlider, SLOT(setValue(int)));

    connect(method1, SIGNAL(toggled(bool)), this, SLOT(setEnableBox(bool)));

    if ( method1->isChecked())
        snapingBorderValue->setValue(50);

    advancedOptions->setHidden(false);
}

void QInciseOperation::setEnableBox(bool i)
{
    advancedOptions->setVisible(i);
}

void QInciseOperation::setFinishIncision(bool i)
{
    finishIncision = i;
}

void QInciseOperation::setkeepPoint(bool i)
{
    keepPoint = i;
}



int QInciseOperation::getIncisionMethod() const
{
    if (method2->isChecked())
        return 1;
    else
        return 0;
}

int QInciseOperation::getSnapingBorderValue() const
{
    return snapingBorderValue->value();
}

int QInciseOperation::getSnapingValue() const
{
    return snapingValue->value();
}

//*******************************************************************************************



//*******************************************************************************************
QTopologyOperation::QTopologyOperation()
{
    //Building the GUI for Topological Operation

    QVBoxLayout *layout=new QVBoxLayout(this);

    // First part: selection of topological operation:
    QHBoxLayout *HLayout1 = new QHBoxLayout();
    QLabel *label1=new QLabel(QString("Topological operation: "), this);
    operationChoice = new QComboBox(this);
    operationChoice->addItem("Remove one element");
    operationChoice->addItem("Remove a zone of elements");

    HLayout1->addWidget (label1);
    HLayout1->addWidget (operationChoice);


    // Second part: advanced settings
    advancedOptions = new QGroupBox(tr("Advanced settings"),this);
    QVBoxLayout *VLayout1 = new QVBoxLayout(advancedOptions);

    // First setting: type of mesh, either surface or volume
    QHBoxLayout *HLayout2 = new QHBoxLayout();

    QLabel *label2 = new QLabel(QString("Remove area type: "), this);
    meshType1 = new QRadioButton(tr("&Surface"), advancedOptions);
    meshType2 = new QRadioButton(tr("&Volume"), advancedOptions);
    meshType1->setChecked (true);

    HLayout2->addWidget (label2);
    HLayout2->addWidget (meshType1);
    HLayout2->addWidget (meshType2);
    VLayout1->addLayout (HLayout2);

    // Second setting: selector scale
    QHBoxLayout *HLayout3 = new QHBoxLayout();

    QLabel *label3=new QLabel(QString("Selector scale: "), this);
    scaleSlider = new QSlider (Qt::Horizontal, this);
    scaleValue = new QSpinBox(this);
    scaleValue->setMinimum(0);
    scaleValue->setMaximum(100);
    scaleValue->setSingleStep(1);
    scaleValue->setEnabled(true);

    HLayout3->addWidget (label3);
    HLayout3->addWidget (scaleSlider);
    HLayout3->addWidget (scaleValue);
    VLayout1->addLayout (HLayout3);


    // Creating UI
    layout->addLayout (HLayout1);
    layout->addWidget(advancedOptions);

    connect(scaleSlider,SIGNAL(valueChanged(int)), scaleValue, SLOT(setValue(int)));
    connect(scaleValue,SIGNAL(valueChanged(int)), scaleSlider, SLOT(setValue(int)));
    connect(operationChoice, SIGNAL(activated(int)), this, SLOT(setEnableBox(int)));

    scaleValue->setValue(0);
    advancedOptions->setHidden(true);

}


double QTopologyOperation::getScale() const
{
    return scaleValue->value();
}

int QTopologyOperation::getTopologicalOperation() const
{
    return operationChoice->currentIndex();
}

bool QTopologyOperation::getVolumicMesh() const
{
    if (meshType2->isChecked())
        return 1;
    else
        return 0;
}

void QTopologyOperation::setEnableBox(int i)
{
    switch (i)
    {
    case 0:
        advancedOptions->setHidden(true);
        break;
    case 1:
        advancedOptions->setHidden(false);
        break;
    default:
        break;
    }
}

//*******************************************************************************************


//*******************************************************************************************
QAddSutureOperation::QAddSutureOperation()
{
    //Building the GUI for the Suture Operation
    QVBoxLayout *layout=new QVBoxLayout(this);

    QHBoxLayout *option1=new QHBoxLayout();
    QLabel *label1=new QLabel(QString("Spring stiffness"), this);
    stiffness = new QLineEdit(QString("10.0"), this);
    option1->addWidget(label1);
    option1->addWidget(stiffness);

    QHBoxLayout *option2=new QHBoxLayout();
    QLabel *label2=new QLabel(QString("Spring damping"), this);
    damping = new QLineEdit(QString("1.0"), this);
    option1->addWidget(label2);
    option1->addWidget(damping);

    layout->addLayout(option1);
    layout->addLayout(option2);
}

double QAddSutureOperation::getStiffness() const
{
    return stiffness->displayText().toDouble();
    //return atof(stiffness->displayText().toStdString().c_str());
}

double QAddSutureOperation::getDamping() const
{
    return damping->displayText().toDouble();
    //return atof(damping->displayText().toStdString());
}

//*******************************************************************************************

} // namespace sofa
} // namespace gui
} // namespace qt


