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

#include <sofa/gui/qt/QMouseOperations.h>
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
    value=new QLineEdit(QString("1000.0"), this);

    layout->addWidget(label);
    layout->addWidget(value);
}

double QAttachOperation::getStiffness() const
{
    return atof(value->displayText().ascii());
}

//*******************************************************************************************


//*******************************************************************************************
QInciseOperation::QInciseOperation()
{
    //Building the GUI for the Injection Operation
    QVBoxLayout *layout=new QVBoxLayout(this);

    // First group box for incision method choice
    incisionMethodChoiceGroup = new QGroupBox(tr("Incision method choice"),this);
    QVBoxLayout *vbox1 = new QVBoxLayout(incisionMethodChoiceGroup);

    method1 = new QRadioButton(tr("&Throw segment: Incise from click to click."), incisionMethodChoiceGroup);
    method2 = new QRadioButton(tr("&Continually: Incise continually from first click localization."), incisionMethodChoiceGroup);
    method1->setChecked (true);

    vbox1->addWidget(method1);
    vbox1->addWidget(method2);

    // Second group box for advanced settings (only snping % value for the moment)
    advancedOptions = new QGroupBox(tr("Advanced settings"),this);
    QVBoxLayout *vbox2 = new QVBoxLayout(advancedOptions);

    // first slider for border snaping
    QHBoxLayout *slider1=new QHBoxLayout();
    QLabel *label1=new QLabel(QString("Distance to snap from border (in %)"), this);
    snapingBorderSlider=new QSlider(Qt::Horizontal, this);
    snapingBorderValue=new QSpinBox(0,100,1,this);
    snapingBorderValue->setEnabled(true);

    slider1->addWidget (label1);
    slider1->addWidget (snapingBorderSlider);
    slider1->addWidget (snapingBorderValue);
    vbox2->addLayout (slider1);

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
    vbox2->addLayout (slider2);


    // Creating UI
    layout->addWidget(incisionMethodChoiceGroup);
    layout->addWidget(advancedOptions);

    connect(snapingBorderSlider,SIGNAL(valueChanged(int)), snapingBorderValue, SLOT(setValue(int)));
    connect(snapingBorderValue,SIGNAL(valueChanged(int)), snapingBorderSlider, SLOT(setValue(int)));

    connect(snapingSlider,SIGNAL(valueChanged(int)), snapingValue, SLOT(setValue(int)));
    connect(snapingValue,SIGNAL(valueChanged(int)), snapingSlider, SLOT(setValue(int)));

    snapingBorderValue->setValue(50);
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
    operationChoice->setObjectName(QString::fromUtf8("comboBox"));
    operationChoice->insertItems(0, QStringList()
            << QApplication::translate("Dialog", "Remove one element", 0, QApplication::UnicodeUTF8)
            << QApplication::translate("Dialog", "Remove a zone of elements", 0, QApplication::UnicodeUTF8));

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

    scaleValue->setValue(0);
    //	operationChoice->setCurrentIndex ( 0 ); ?

    /*if (operationChoice->currentIndex() == 0)
      advancedOptions->setEnabled (false);
    else
    advancedOptions->setEnabled (true);*/
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

//*******************************************************************************************


//*******************************************************************************************
QInjectOperation::QInjectOperation()
{
    //Building the GUI for the Injection Operation
    QHBoxLayout *layout=new QHBoxLayout(this);
    QLabel *label1=new QLabel(QString("Potential Value"), this);
    value=new QLineEdit(QString("100.0"), this);

    QLabel *label2=new QLabel(QString("State Tag"), this);
    tag=new QLineEdit(QString("elec"), this);

    layout->addWidget(label1);
    layout->addWidget(value);

    layout->addWidget(label2);
    layout->addWidget(tag);
}

double QInjectOperation::getPotentialValue() const
{
    return atof(value->displayText().ascii());
}

std::string QInjectOperation::getStateTag() const
{
    return (std::string)(tag->displayText()).ascii();
}

//*******************************************************************************************

} // namespace sofa
} // namespace gui
} // namespace qt


