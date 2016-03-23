/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef MATERIAL_DATAWIDGET_H
#define MATERIAL_DATAWIDGET_H
#include "DataWidget.h"
#include <sofa/core/loader/Material.h>
#include <sofa/defaulttype/Vec.h>

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

namespace sofa
{
namespace gui
{
namespace qt
{

class RGBAColorPicker : public QWidget
{
    Q_OBJECT
signals:
    void hasChanged();
public:
    RGBAColorPicker(QWidget* parent);
    void setColor( const sofa::defaulttype::Vec4f& color );
    sofa::defaulttype::Vec4f getColor() const;
protected:
    QRgb _rgba;
    QLineEdit* _r;
    QLineEdit* _g;
    QLineEdit* _b;
    QLineEdit* _a;
    QPushButton* _colorButton;

protected slots:
    void updateRGBAColor();
    void redrawColorButton();
    void raiseQColorDialog();
};

class MaterialDataWidget : public TDataWidget<sofa::core::loader::Material>
{
    Q_OBJECT
public:
    MaterialDataWidget(QWidget* parent,
            const char* name,
            core::objectmodel::Data<sofa::core::loader::Material>* data):
        TDataWidget<sofa::core::loader::Material>(parent,name,data)
    {}

    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);
    virtual unsigned int numColumnWidget() {return 1;}

protected:
    virtual void readFromData();
    virtual void writeToData();
    QLineEdit* _nameEdit;
    RGBAColorPicker* _ambientPicker;
    RGBAColorPicker* _emissivePicker;
    RGBAColorPicker* _specularPicker;
    RGBAColorPicker* _diffusePicker;
    QLineEdit*  _shininessEdit;
    QCheckBox* _ambientCheckBox;
    QCheckBox* _emissiveCheckBox;
    QCheckBox* _specularCheckBox;
    QCheckBox* _diffuseCheckBox;
    QCheckBox* _shininessCheckBox;
};


typedef helper::vector<sofa::core::loader::Material> VectorMaterial;
class VectorMaterialDataWidget : public TDataWidget< VectorMaterial >
{
    Q_OBJECT
public:
    VectorMaterialDataWidget(QWidget* parent,
            const char* name,
            core::objectmodel::Data< helper::vector<sofa::core::loader::Material> >* data):
        TDataWidget< helper::vector<sofa::core::loader::Material> >(parent,name,data),
        _materialDataWidget(NULL),
        _currentMaterial(0,data->isDisplayed(),data->isReadOnly()),
        _comboBox(NULL)
    {

    };

    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);
    virtual unsigned int numColumnWidget() {return 1;}


protected:
    virtual void readFromData();
    virtual void writeToData();

    MaterialDataWidget* _materialDataWidget;
    VectorMaterial _vectorEditedMaterial;
    core::objectmodel::Data<sofa::core::loader::Material> _currentMaterial;
    QComboBox* _comboBox;
    int _currentMaterialPos;
protected slots:
    void changeMaterial( int );
};
}
}


}

#endif

