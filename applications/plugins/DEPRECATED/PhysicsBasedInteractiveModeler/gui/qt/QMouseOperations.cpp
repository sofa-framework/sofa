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

#include "QMouseOperations.h"
#include <PhysicsBasedInteractiveModeler/pim/SculptBodyPerformer.h>
#ifdef SOFA_QT4
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QRadioButton>
#include <QPushButton>
#else
#include <qlayout.h>
#include <qlabel.h>
#include <qradiobutton.h>
#include <qpushbutton.h>
#endif

namespace plugins
{
namespace pim
{
namespace gui
{
namespace qt
{

using namespace sofa::defaulttype;

QSculptOperation::QSculptOperation()
{

    //Building the GUI for Sculpt Operation
    QVBoxLayout *layout=new QVBoxLayout(this);

    options = new QGroupBox(QString("Options"),this);
    VLayout = new QVBoxLayout(options);


    QHBoxLayout *HLayout = new QHBoxLayout();
    inflateRadioButton = new QRadioButton(QString("Inflate"), options);
    inflateRadioButton->setChecked(true);
    deflateRadioButton = new QRadioButton(QString("Deflate"), options);
    fixRadioButton = new QRadioButton(QString("Fix"), options);

    HLayout->addWidget(inflateRadioButton);
    HLayout->addWidget(deflateRadioButton);
    HLayout->addWidget(fixRadioButton);
    VLayout->addLayout(HLayout);

    QHBoxLayout *HLayout1 = new QHBoxLayout();
    QLabel *scaleLabel=new QLabel(QString("Scale"), this);
    scaleSlider=new QSlider(Qt::Horizontal, this);
    scaleValue=new QSpinBox(0,100,1,this);
    scaleValue->setEnabled(true);

    HLayout1->addWidget(scaleLabel);
    HLayout1->addWidget(scaleSlider);
    HLayout1->addWidget(scaleValue);
    VLayout->addLayout(HLayout1);

    HLayout2 = new QHBoxLayout();
    forceLabel=new QLabel(QString("Force"), this);
    forceSlider=new QSlider(Qt::Horizontal, this);
    forceValue=new QSpinBox(0,100,1,this);
    forceValue->setEnabled(true);

    HLayout2->addWidget(forceLabel);
    HLayout2->addWidget(forceSlider);
    HLayout2->addWidget(forceValue);
    VLayout->addLayout(HLayout2);

    HLayout3 = new QHBoxLayout();
    massLabel=new QLabel(QString("Mass"), this);
    massValue=new QLineEdit(this);
    stiffnessLabel=new QLabel(QString("Stiffness"), this);
    stiffnessValue=new QLineEdit(this);
    dampingLabel=new QLabel(QString("Damping"), this);
    dampingValue=new QLineEdit(this);

    HLayout3->addWidget(massLabel);
    HLayout3->addWidget(massValue);
    HLayout3->addWidget(stiffnessLabel);
    HLayout3->addWidget(stiffnessValue);
    HLayout3->addWidget(dampingLabel);
    HLayout3->addWidget(dampingValue);

    VLayout->addLayout(HLayout3);

    QHBoxLayout *HLayout3 = new QHBoxLayout();
    animatePushButton = new QPushButton(QString("Animate"), options);
    animatePushButton->setMaximumSize(75,30);

    HLayout3->addWidget(animatePushButton);
    VLayout->addLayout(HLayout3);

    layout->addWidget(options);

    connect(forceSlider,SIGNAL(valueChanged(int)), forceValue, SLOT(setValue(int)));
    connect(scaleSlider,SIGNAL(valueChanged(int)), scaleValue, SLOT(setValue(int)));

    connect(scaleSlider,SIGNAL(valueChanged(int)), this, SLOT(setScale()));
    /* Add solver, mass and forcefield to simulate added materia */
    connect(animatePushButton,SIGNAL(clicked()), this, SLOT(animate()));

    connect(fixRadioButton,SIGNAL(toggled(bool)), this, SLOT(updateInterface(bool)));

    forceSlider->setValue(60);
    scaleSlider->setValue(70);

    massValue->insert("10");
    stiffnessValue->insert("100");
    dampingValue->insert("0.2");
}

void QSculptOperation::updateInterface(bool checked)
{
    if (!checked)
    {
        forceLabel->setHidden(false);
        forceSlider->setHidden(false);
        forceValue->setHidden(false);
        animatePushButton->setHidden(false);
    }
    else
    {
        forceLabel->setHidden(true);
        forceSlider->setHidden(true);
        forceValue->setHidden(true);
        animatePushButton->setHidden(true);
    }
}

double QSculptOperation::getForce() const
{
    return forceValue->value();
}

double QSculptOperation::getScale() const
{
    return scaleValue->value();
}

double QSculptOperation::getMass() const
{
    return (massValue->text()).toDouble();
}

double QSculptOperation::getStiffness() const
{
    return (stiffnessValue->text()).toDouble();
}

double QSculptOperation::getDamping() const
{
    return (dampingValue->text()).toDouble();
}

bool QSculptOperation::isCheckedInflate() const
{
    return inflateRadioButton->isChecked();
}

bool QSculptOperation::isCheckedDeflate() const
{
    return deflateRadioButton->isChecked();
}

bool QSculptOperation::isCheckedFix() const
{
    return fixRadioButton->isChecked();
}

void QSculptOperation::setScale()
{
    if (performer == NULL) return;

    SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<SculptBodyPerformerConfiguration*>(performer);

    if (performerConfiguration == NULL) return;

    performerConfiguration->setScale(getScale());
}

void QSculptOperation::animate()
{
    if (performer == NULL) return;

    SculptBodyPerformer<Vec3Types>* sculptPerformer=dynamic_cast<SculptBodyPerformer<Vec3Types>*>(performer);
    sculptPerformer->animate();
}

} // namespace qt
} // namespace gui
} // namespace pim
} // namespace plugins
