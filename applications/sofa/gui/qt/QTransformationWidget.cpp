/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "QTransformationWidget.h"
#include <SofaSimulationCommon/TransformationVisitor.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QLabel>

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif


namespace sofa
{

namespace gui
{
namespace qt
{
QTransformationWidget::QTransformationWidget(QWidget* parent, QString name):QGroupBox(parent), numWidgets_(2)

{
    //this->setColumns(4);

    this->setTitle(name);
    //********************************************************************************
    //Translation
    new QLabel(QString("Translation"), this);
    translation[0] = new WDoubleLineEdit( this, "translation[0]" );
    translation[0]->setMinValue( (double)-INFINITY );
    translation[0]->setMaxValue( (double)INFINITY );

    translation[1] = new WDoubleLineEdit( this, "translation[1]" );
    translation[1]->setMinValue( (double)-INFINITY );
    translation[1]->setMaxValue( (double)INFINITY );

    translation[2] = new WDoubleLineEdit( this, "translation[2]" );
    translation[2]->setMinValue( (double)-INFINITY );
    translation[2]->setMaxValue( (double)INFINITY );


    //********************************************************************************
    //Rotation
    new QLabel(QString("Rotation"), this);
    rotation[0] = new WDoubleLineEdit( this, "rotation[0]" );
    rotation[0]->setMinValue( (double)-INFINITY );
    rotation[0]->setMaxValue( (double)INFINITY );

    rotation[1] = new WDoubleLineEdit( this, "rotation[1]" );
    rotation[1]->setMinValue( (double)-INFINITY );
    rotation[1]->setMaxValue( (double)INFINITY );

    rotation[2] = new WDoubleLineEdit( this, "rotation[2]" );
    rotation[2]->setMinValue( (double)-INFINITY );
    rotation[2]->setMaxValue( (double)INFINITY );


    //********************************************************************************
    //Scale
    QLabel *textScale = new QLabel(QString("Scale"), this);
    scale[0] = new WDoubleLineEdit( this, "scale[0]" );
    scale[0]->setMinValue( (double)-INFINITY );
    scale[0]->setMaxValue( (double)INFINITY );

    scale[1] = new WDoubleLineEdit( this, "scale[1]" );
    scale[1]->setMinValue( (double)-INFINITY );
    scale[1]->setMaxValue( (double)INFINITY );

    scale[2] = new WDoubleLineEdit( this, "scale[2]" );
    scale[2]->setMinValue( (double)-INFINITY );
    scale[2]->setMaxValue( (double)INFINITY );

    setDefaultValues();

    connect( translation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( translation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( translation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( rotation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( rotation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( rotation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( scale[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( scale[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( scale[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );


    //Option still experimental : disabled !!!!
    textScale->hide();
    scale[0]->hide();
    scale[1]->hide();
    scale[2]->hide();
}

void QTransformationWidget::setDefaultValues()
{
    //********************************************************************************
    //Default values
    translation[0]->setValue(0);
    translation[1]->setValue(0);
    translation[2]->setValue(0);

    rotation[0]->setValue(0);
    rotation[1]->setValue(0);
    rotation[2]->setValue(0);

    scale[0]->setValue(1);
    scale[1]->setValue(1);
    scale[2]->setValue(1);
}

bool QTransformationWidget::isDefaultValues() const
{
    return ( (translation[0]->getValue() == 0 && translation[1]->getValue() == 0 && translation[2]->getValue() == 0 ) &&
            (rotation[0]->getValue() == 0    && rotation[1]->getValue() == 0    && rotation[2]->getValue() == 0 ) &&
            (scale[0]->getValue() == 1       && scale[1]->getValue() == 1       && scale[2]->getValue() == 1 ) );
}

void QTransformationWidget::applyTransformation(simulation::Node *node)
{
    sofa::simulation::TransformationVisitor transform(sofa::core::ExecParams::defaultInstance());
    transform.setTranslation(translation[0]->getValue(),translation[1]->getValue(),translation[2]->getValue());
    transform.setRotation(rotation[0]->getValue(),rotation[1]->getValue(),rotation[2]->getValue());
    transform.setScale(scale[0]->getValue(),scale[1]->getValue(),scale[2]->getValue());
    transform.execute(node);
}


} // qt
} //gui
} //sofa

