/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include "DataWidget.h"
#include <sofa/type/Material.h>
#include <sofa/type/Vec.h>

#include <QColorDialog>
#include <QPainter>
#include <QStyle>
#include <QCheckBox>
#include <QComboBox>
#include <QColor>
#include <QPixmap>
#include <QLineEdit>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QValidator>
#include <QColorDialog>

#include "QRGBAColorPicker.h"

/// Private namespace
namespace sofa::gui::qt::materialdatawidget_h
{

using sofa::gui::qt::QRGBAColorPicker ;
using sofa::type::Material ;
using sofa::core::objectmodel::Data ;

class MaterialDataWidget : public TDataWidget<Material>
{
    Q_OBJECT
public:
    MaterialDataWidget(QWidget* parent,
                       const char* name,
                       Data<Material>* data):
        TDataWidget<Material>(parent,name,data)
    {}

    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);
    virtual unsigned int numColumnWidget() {return 1;}

protected:
    virtual void readFromData();
    virtual void writeToData();
    QLineEdit* _nameEdit;
    QRGBAColorPicker* _ambientPicker;
    QRGBAColorPicker* _emissivePicker;
    QRGBAColorPicker* _specularPicker;
    QRGBAColorPicker* _diffusePicker;
    QLineEdit*  _shininessEdit;
    QCheckBox* _ambientCheckBox;
    QCheckBox* _emissiveCheckBox;
    QCheckBox* _specularCheckBox;
    QCheckBox* _diffuseCheckBox;
    QCheckBox* _shininessCheckBox;
};


typedef type::vector<Material> VectorMaterial;
class VectorMaterialDataWidget : public TDataWidget< VectorMaterial >
{
    Q_OBJECT
public:
    VectorMaterialDataWidget(QWidget* parent,
                             const char* name,
                             Data< type::vector<Material> >* data):
        TDataWidget< type::vector<Material> >(parent,name,data),
        _materialDataWidget(nullptr),
        _currentMaterial(0,data->isDisplayed(),data->isReadOnly()),
        _comboBox(nullptr)
    {

    }

    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);
    virtual unsigned int numColumnWidget() {return 1;}


protected:
    virtual void readFromData();
    virtual void writeToData();

    MaterialDataWidget* _materialDataWidget;
    VectorMaterial _vectorEditedMaterial;
    core::objectmodel::Data<Material> _currentMaterial;
    QComboBox* _comboBox;
    int _currentMaterialPos;

protected slots:
    void changeMaterial( int );
};

} // namespace sofa::gui::qt::materialdatawidget_h

namespace sofa::gui::qt
{
    using sofa::gui::qt::materialdatawidget_h::MaterialDataWidget;
} // namespace sofa::gui::qt
