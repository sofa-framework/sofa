/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/gui/qt/QMouseOperations.h>
#include <sofa/gui/qt/QDisplayDataWidget.h>
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
//*******************************************************************************************
QAttachOperation::QAttachOperation()
{
    //Building the GUI for the Attach Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label=new QLabel(QString("Stiffness"), this);

    {
        //Stiffness
        DataWidget::CreatorArgument argStiffness;
        argStiffness.data = setting.getDataStiffness();
        argStiffness.name =  argStiffness.data->getName();
        argStiffness.parent = this;
        argStiffness.readOnly = argStiffness.data->isReadOnly();
        stiffnessWidget = DataWidget::CreateDataWidget(argStiffness);
        connect(stiffnessWidget, SIGNAL(WidgetDirty(bool)), this, SLOT(WidgetDirty(bool)));
    }


    QLabel *labelSize=new QLabel(QString("Arrow Size"), this);
    {
        //Arrow Size
        DataWidget::CreatorArgument argArrowSize;
        argArrowSize.data = setting.getDataArrowSize();
        argArrowSize.name =  argArrowSize.data->getName();
        argArrowSize.parent = this;
        argArrowSize.readOnly = argArrowSize.data->isReadOnly();
        arrowSizeWidget = DataWidget::CreateDataWidget(argArrowSize);
        connect(arrowSizeWidget, SIGNAL(WidgetDirty(bool)), this, SLOT(WidgetDirty(bool)));
    }

    layout->addWidget(stiffnessWidget);
    layout->addWidget(label);

//        layout->addWidget(labelSize);
//        layout->addWidget(arrowSizeWidget);
}

void QAttachOperation::WidgetDirty(bool b)
{
    if (b)
    {
        DataWidget *dataW=(DataWidget*) sender();
        dataW->updateDataValue();
    }
}

//      double QAttachOperation::getStiffness() const
//      {
//        return atof(value->displayText().ascii());
//      }

//      double QAttachOperation::getArrowSize() const
//      {
//        return atof(size->displayText().ascii());
//      }

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
    snapingBorderValue=new QSpinBox(0,100,1,this);
    snapingBorderValue->setEnabled(true);

    slider1->addWidget (label1);
    slider1->addWidget (snapingBorderSlider);
    slider1->addWidget (snapingBorderValue);
    vbox3->addLayout (slider1);

    // second slider for along path snaping
    QHBoxLayout *slider2=new QHBoxLayout();
    QLabel *label2=new QLabel(QString("Distance to snap along path (in %)"), this);
    snapingSlider=new QSlider(Qt::Horizontal, this);
    snapingValue=new QSpinBox(0,100,1,this);
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
    advancedOptions->setShown(i);
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
QFixOperation::QFixOperation()
{
    //Building the GUI for the Fix Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label=new QLabel(QString("Fixation"), this);
    value=new QLineEdit(QString("10000.0"), this);

    layout->addWidget(label);
    layout->addWidget(value);
}

double QFixOperation::getStiffness() const
{
    return atof(value->displayText().ascii());
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
    operationChoice->insertItem("Remove one element");
    operationChoice->insertItem("Remove a zone of elements");

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
    scaleValue = new QSpinBox(0,100,1,this);
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
#ifdef SOFA_QT4
    return operationChoice->currentIndex();
#else
    return operationChoice->currentItem();
#endif
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
    return atof(stiffness->displayText().ascii());
}

double QAddSutureOperation::getDamping() const
{
    return atof(damping->displayText().ascii());
}

//*******************************************************************************************

} // namespace sofa
} // namespace gui
} // namespace qt


