/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_IMAGE_IMAGETRANSFORMWIDGET_CPP

#include "ImageTransformWidget.h"
#include <sofa/helper/Factory.inl>
#include <iostream>


namespace sofa
{

namespace gui
{

namespace qt
{

using namespace defaulttype;

template class SOFA_IMAGE_GUI_API TDataWidget<ImageLPTransform<double> >;
template class SOFA_IMAGE_GUI_API TDataWidget<ImageLPTransform<float> >;

helper::Creator<DataWidgetFactory, ImageLPTransformWidget< ImageLPTransform<double> > > DWClass_imageTd("default",true);
helper::Creator<DataWidgetFactory, ImageLPTransformWidget< ImageLPTransform<float> > > DWClass_imageTf("default",true);

template<class TransformType>
bool ImageLPTransformWidget<TransformType>::createWidgets()
{
    QGridLayout* layout = new QGridLayout(this);

    //********************************************************************************
    //Translation
    translationL = new QLabel(QString("Translation"), this);
    layout->addWidget(translationL,0,0);

    translation[0] = new WDoubleLineEdit( this, "translation[0]" );
    translation[0]->setMinValue( (double)-INFINITY );
    translation[0]->setMaxValue( (double)INFINITY );
    layout->addWidget(translation[0],0,1);

    translation[1] = new WDoubleLineEdit( this, "translation[1]" );
    translation[1]->setMinValue( (double)-INFINITY );
    translation[1]->setMaxValue( (double)INFINITY );
    layout->addWidget(translation[1],0,2);

    translation[2] = new WDoubleLineEdit( this, "translation[2]" );
    translation[2]->setMinValue( (double)-INFINITY );
    translation[2]->setMaxValue( (double)INFINITY );
    layout->addWidget(translation[2],0,3);

    //********************************************************************************
    //Rotation
    rotationL = new QLabel(QString("Rotation"), this);
    layout->addWidget(rotationL,1,0);

    rotation[0] = new WDoubleLineEdit( this, "rotation[0]" );
    rotation[0]->setMinValue( (double)-INFINITY );
    rotation[0]->setMaxValue( (double)INFINITY );
    layout->addWidget(rotation[0],1,1);

    rotation[1] = new WDoubleLineEdit( this, "rotation[1]" );
    rotation[1]->setMinValue( (double)-INFINITY );
    rotation[1]->setMaxValue( (double)INFINITY );
    layout->addWidget(rotation[1],1,2);

    rotation[2] = new WDoubleLineEdit( this, "rotation[2]" );
    rotation[2]->setMinValue( (double)-INFINITY );
    rotation[2]->setMaxValue( (double)INFINITY );
    layout->addWidget(rotation[2],1,3);


    //********************************************************************************
    //Scale

    scaleL =new QLabel(QString("Scale3d"), this);
    layout->addWidget(scaleL,2,0);

    scale[0] = new WDoubleLineEdit( this, "scale[0]" );
    scale[0]->setMinValue( (double)-INFINITY );
    scale[0]->setMaxValue( (double)INFINITY );
    layout->addWidget(scale[0],2,1);

    scale[1] = new WDoubleLineEdit( this, "scale[1]" );
    scale[1]->setMinValue( (double)-INFINITY );
    scale[1]->setMaxValue( (double)INFINITY );
    layout->addWidget(scale[1],2,2);

    scale[2] = new WDoubleLineEdit( this, "scale[2]" );
    scale[2]->setMinValue( (double)-INFINITY );
    scale[2]->setMaxValue( (double)INFINITY );
    layout->addWidget(scale[2],2,3);


    ////********************************************************************************
    //offsetT

    offsetTL = new QLabel(QString("Time offset"), this);
    layout->addWidget(offsetTL,3,0);

    offsetT = new WDoubleLineEdit( this, "offsetT" );
    offsetT->setMinValue( (double)-INFINITY );
    offsetT->setMaxValue( (double)INFINITY );
    layout->addWidget(offsetT,3,1);


    ////********************************************************************************
    //scaleT

    scaleTL = new QLabel(QString("Time scale"), this);
    layout->addWidget(scaleTL,4,0);

    scaleT = new WDoubleLineEdit( this, "scaleT" );
    scaleT->setMinValue( (double)-INFINITY );
    scaleT->setMaxValue( (double)INFINITY );
    layout->addWidget(scaleT,4,1);

    ////********************************************************************************
    //isPerspective

    isPerspectiveL = new QLabel(QString("isPerspective"), this);
    layout->addWidget(isPerspectiveL,5,0);

    isPerspective = new QCheckBox( this );
    layout->addWidget(isPerspective,5,1);


    this->connect( translation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( translation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( translation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( rotation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( rotation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( rotation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( scale[0], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( scale[1], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( scale[2], SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( offsetT, SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( scaleT, SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    this->connect( isPerspective, SIGNAL( stateChanged(int) ), this, SLOT( setWidgetDirty() ) );

    return true;
}

template<class TransformType>
void ImageLPTransformWidget<TransformType>::readFromData()
{
    helper::ReadAccessor<MyTData> ra (this->getData());

    translation[0]->setValue(ra->getTranslation()[0]);
    translation[1]->setValue(ra->getTranslation()[1]);
    translation[2]->setValue(ra->getTranslation()[2]);
    rotation[0]->setValue(ra->getRotation()[0]);
    rotation[1]->setValue(ra->getRotation()[1]);
    rotation[2]->setValue(ra->getRotation()[2]);
    scale[0]->setValue(ra->getScale()[0]);
    scale[1]->setValue(ra->getScale()[1]);
    scale[2]->setValue(ra->getScale()[2]);
    offsetT->setValue(ra->getOffsetT());
    scaleT->setValue(ra->getScaleT());
    isPerspective->setChecked(ra->isPerspective()!=0);

    this->setWidgetDirty(false);
}

template<class TransformType>
void ImageLPTransformWidget<TransformType>::writeToData()
{
    helper::WriteOnlyAccessor<MyTData> wa (this->getData());

    wa->getTranslation()[0]=translation[0]->getValue();
    wa->getTranslation()[1]=translation[1]->getValue();
    wa->getTranslation()[2]=translation[2]->getValue();
    wa->getRotation()[0]=rotation[0]->getValue();
    wa->getRotation()[1]=rotation[1]->getValue();
    wa->getRotation()[2]=rotation[2]->getValue();
    wa->getScale()[0]=scale[0]->getValue();
    wa->getScale()[1]=scale[1]->getValue();
    wa->getScale()[2]=scale[2]->getValue();
    wa->getOffsetT()=offsetT->getValue();
    wa->getScaleT()=scaleT->getValue();
    wa->isPerspective()=(int)isPerspective->isChecked();
    wa->update();
}

} // qt
} // gui
} // sofa


