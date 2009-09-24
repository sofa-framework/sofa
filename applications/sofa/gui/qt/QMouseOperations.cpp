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
#ifdef SOFA_DEV
#include <sofa/component/collision/SculptBodyPerformer.h>
#endif
#ifdef SOFA_QT4
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#else
#include <qlayout.h>
#include <qlabel.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{
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



QSculptOperation::QSculptOperation()
{
    QGridLayout *layout=new QGridLayout(this,2,3);
    QLabel *forceLabel=new QLabel(QString("Force"), this);
    forceSlider=new QSlider(Qt::Horizontal, this);
    forceValue=new QSpinBox(0,100,1,this);
    forceValue->setEnabled(false);


    layout->addWidget(forceLabel,0,0);
    layout->addWidget(forceSlider,0,1);
    layout->addWidget(forceValue,0,2);

    QLabel *scaleLabel=new QLabel(QString("Scale"), this);
    scaleSlider=new QSlider(Qt::Horizontal, this);
    scaleValue=new QSpinBox(0,100,1,this);
    scaleValue->setEnabled(false);

    layout->addWidget(scaleLabel,1,0);
    layout->addWidget(scaleSlider,1,1);
    layout->addWidget(scaleValue,1,2);


    connect(forceSlider,SIGNAL(valueChanged(int)), forceValue, SLOT(setValue(int)));
    connect(scaleSlider,SIGNAL(valueChanged(int)), scaleValue, SLOT(setValue(int)));
    connect(scaleSlider,SIGNAL(valueChanged(int)), this, SLOT(setScale()));

    forceSlider->setValue(50);
    scaleSlider->setValue(50);
}

double QSculptOperation::getForce() const
{
    return forceValue->value();
}

double QSculptOperation::getScale() const
{
    return scaleValue->value();
}

void QSculptOperation::setScale()
{
#ifdef SOFA_DEV
    if (!performer) return;
    component::collision::SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::SculptBodyPerformerConfiguration*>(performer);
    if (!performerConfiguration) return;
    performerConfiguration->setScale(getScale());
#endif
}

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
}
}
}
