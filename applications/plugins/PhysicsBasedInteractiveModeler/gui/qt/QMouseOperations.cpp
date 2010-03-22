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

#include "QMouseOperations.h"
#include <plugins/PhysicsBasedInteractiveModeler/pim/SculptBodyPerformer.h>
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
    sculptRadioButton = new QRadioButton(QString("Sculpt"), options);
    sculptRadioButton->setChecked(true);
    fixRadioButton = new QRadioButton(QString("Fix"), options);

    HLayout->addWidget(sculptRadioButton);
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

    QHBoxLayout *HLayout3 = new QHBoxLayout();
    animatePushButton = new QPushButton(QString("Animate"), options);
    animatePushButton->setMaximumSize(75,30);

#ifdef SOFA_QT4
    animatePushButton->setCheckable(true);
#else
    animatePushButton->setToggleButton(true);
#endif
    HLayout3->addWidget(animatePushButton);
    VLayout->addLayout(HLayout3);

    layout->addWidget(options);

    connect(forceSlider,SIGNAL(valueChanged(int)), forceValue, SLOT(setValue(int)));
    connect(scaleSlider,SIGNAL(valueChanged(int)), scaleValue, SLOT(setValue(int)));

    connect(scaleSlider,SIGNAL(valueChanged(int)), this, SLOT(setScale()));
    /* Add solver, mass and forcefield to simulate added materia */
    connect(animatePushButton,SIGNAL(toggled(bool)), this, SLOT(animate(bool)));

    connect(fixRadioButton,SIGNAL(toggled(bool)), this, SLOT(updateInterface(bool)));

    forceSlider->setValue(10);
    scaleSlider->setValue(70);
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

void QSculptOperation::animate(bool checked)
{
    animated = checked;

    if (performer == NULL) return;

    SculptBodyPerformer<Vec3Types>* sculptPerformer=dynamic_cast<SculptBodyPerformer<Vec3Types>*>(performer);
    sculptPerformer->animate(checked);
}

} // namespace qt
} // namespace gui
} // namespace pim
} // namespace plugins
