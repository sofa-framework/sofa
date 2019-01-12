/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_IMAGE_IMAGETRANSFORMWIDGET_H
#define SOFA_IMAGE_IMAGETRANSFORMWIDGET_H

#include <image/image_gui/config.h>
#include <sofa/gui/qt/DataWidget.h>
#include <sofa/gui/qt/SimpleDataWidget.h>
#include <sofa/gui/qt/WDoubleLineEdit.h>

#include <QTextEdit>
#include <QGroupBox>
#include <QLabel>
#include <QGridLayout>
#include <QString>
#include <QDoubleSpinBox>

#include "../ImageTypes.h"
#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Data.h>

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif


namespace sofa
{
namespace gui
{
namespace qt
{

template<class _TransformType>
class SOFA_IMAGE_GUI_API ImageLPTransformWidget : public TDataWidget<_TransformType>
{

public :
    typedef _TransformType TransformType;
    typedef typename TransformType::Real Real;
    typedef typename TransformType::Coord Coord;
    typedef TDataWidget<TransformType> Inherited;
    typedef typename Inherited::MyTData MyTData;

    ImageLPTransformWidget(QWidget* parent, const char* name, core::objectmodel::Data<_TransformType>* data)
        : Inherited(parent,name,data)
    {}

    virtual bool createWidgets();
    virtual void setDataReadOnly(bool /*readOnly*/) {};

    virtual unsigned int sizeWidget() {return 12;}
    virtual unsigned int numColumnWidget() {return 3;}

protected:
    virtual void readFromData();
    virtual void writeToData();

protected:

    WDoubleLineEdit* translation[3]; QLabel* translationL;
    WDoubleLineEdit* rotation[3];	 QLabel* rotationL;
    WDoubleLineEdit* scale[3];		 QLabel* scaleL;
    WDoubleLineEdit* offsetT;		 QLabel* offsetTL;
    WDoubleLineEdit* scaleT;		 QLabel* scaleTL;
    QCheckBox* isPerspective;		 QLabel* isPerspectiveL;

};


}

}

}

#endif // SOFA_IMAGE_IMAGETRANSFORMWIDGET_H
