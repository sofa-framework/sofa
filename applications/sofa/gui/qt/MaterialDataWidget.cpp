/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "MaterialDataWidget.h"
#include <limits>

namespace sofa
{
namespace gui
{
namespace qt
{

helper::Creator<DataWidgetFactory,MaterialDataWidget> DWClass_MeshMaterial("default",true);
helper::Creator<DataWidgetFactory,VectorMaterialDataWidget> DWClass_MeshVectorMaterial("default",true);
RGBAColorPicker::RGBAColorPicker(QWidget* parent):QWidget(parent)
{

    _r = new QLineEdit(this);
    _r->setValidator(new QIntValidator(0,255,this));
    _g = new QLineEdit(this);
    _g->setValidator(new QIntValidator(0,255,this));
    _b = new QLineEdit(this);
    _b->setValidator(new QIntValidator(0,255,this));
    _a = new QLineEdit(this);
    _a->setValidator(new QIntValidator(0,255,this));
    _colorButton = new QPushButton(this);
    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->addWidget(_colorButton);
    layout->addWidget(new QLabel("r",this));
    layout->addWidget(_r);
    layout->addWidget(new QLabel("g",this));
    layout->addWidget(_g);
    layout->addWidget(new QLabel("b",this));
    layout->addWidget(_b);
    layout->addWidget(new QLabel("a",this));
    layout->addWidget(_a);
    connect( _r , SIGNAL( textChanged(const QString & ) ) ,this , SIGNAL( hasChanged() ) );
    connect( _g , SIGNAL( textChanged(const QString & ) ) ,this , SIGNAL( hasChanged() ) );
    connect( _b , SIGNAL( textChanged(const QString & ) ) ,this , SIGNAL( hasChanged() ) );
    connect( _a , SIGNAL( textChanged(const QString & ) ) ,this , SIGNAL( hasChanged() ) );
    connect( _r , SIGNAL( returnPressed() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _g , SIGNAL( returnPressed() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _b , SIGNAL( returnPressed() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _a , SIGNAL( returnPressed() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _r , SIGNAL( editingFinished() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _g , SIGNAL( editingFinished() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _b , SIGNAL( editingFinished() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _a , SIGNAL( editingFinished() ) ,this , SLOT( updateRGBAColor() ) );
    connect( _colorButton, SIGNAL( clicked() ), this, SLOT( raiseQColorDialog() ) );

}

defaulttype::Vec4f RGBAColorPicker::getColor() const
{
    typedef unsigned char uchar;
    const uchar max = std::numeric_limits<uchar>::max();
    defaulttype::Vec4f color;
    float r = _r->text().toFloat();
    float g = _g->text().toFloat();
    float b = _b->text().toFloat();
    float a = _a->text().toFloat();
    r /= max;
    g /= max;
    b /= max;
    a /= max;

    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = a;
    return color;



}

void RGBAColorPicker::updateRGBAColor()
{
    typedef unsigned char uchar;
    defaulttype::Vec4f color;
    const uchar r = _r->text().toInt();
    const uchar g = _g->text().toInt();
    const uchar b = _b->text().toInt();
    const uchar a = _a->text().toInt();
    _rgba = qRgba(r,g,b,a);
    redrawColorButton();
}

void RGBAColorPicker::setColor(const sofa::defaulttype::Vec4f& color)
{
    typedef unsigned char uchar;
    const uchar max = std::numeric_limits<uchar>::max();
    const uchar r = uchar(  max * color[0] );
    const uchar g = uchar(  max * color[1] );
    const uchar b = uchar(  max * color[2] );
    const uchar a = uchar(  max * color[3] );
    QString str;
    str.setNum(r);
    _r->setText(str);
    str.setNum(g);
    _g->setText(str);
    str.setNum(b);
    _b->setText(str);
    str.setNum(a);
    _a->setText(str);
    _rgba = qRgba(r,g,b,a);

    redrawColorButton();

}

void RGBAColorPicker::redrawColorButton()
{
    int w=_colorButton->width();
    int h=_colorButton->height();

    QPixmap *pix=new QPixmap(25,20);
    pix->fill(QColor(qRed(_rgba), qGreen(_rgba), qBlue(_rgba)));
    _colorButton->setIcon(QIcon(*pix));

    _colorButton->resize(w,h);
}

void RGBAColorPicker::raiseQColorDialog()
{
    typedef unsigned char uchar;
    const uchar max = std::numeric_limits<uchar>::max();
    int r,g,b,a;
    bool ok;
    defaulttype::Vec4f color;
    QColor qcolor = QColorDialog::getRgba(_rgba,&ok,this);
    if( ok )
    {
        QRgb rgba=qcolor.rgb();
        r=qRed(rgba);
        g=qGreen(rgba);
        b=qBlue(rgba);
        a=qAlpha(rgba);
        color[0] = (float)r / max;
        color[1] = (float)g / max;
        color[2] = (float)b / max;
        color[3] = (float)a / max;
        setColor(color);
        emit hasChanged();
    }
}

bool MaterialDataWidget::createWidgets()
{

    _nameEdit = new QLineEdit(this);
    _ambientPicker = new RGBAColorPicker(this);
    _ambientCheckBox = new QCheckBox(this);
    _emissivePicker = new RGBAColorPicker(this);
    _emissiveCheckBox = new QCheckBox(this);
    _specularPicker = new RGBAColorPicker(this);
    _specularCheckBox = new QCheckBox(this);
    _diffusePicker = new RGBAColorPicker(this);
    _diffuseCheckBox = new QCheckBox(this);
    _shininessEdit = new QLineEdit(this);
    _shininessEdit->setValidator(new QDoubleValidator(this) );
    _shininessCheckBox = new QCheckBox(this);
    QVBoxLayout* layout = new QVBoxLayout(this);

    //QGridLayout* grid = new QGridLayout(5,3);
    QGridLayout* grid = new QGridLayout();
    grid->setSpacing(1);
    QHBoxLayout* hlayout = new QHBoxLayout();
    hlayout->addWidget(new QLabel("Name",this));
    hlayout->addWidget(_nameEdit);

    grid->addWidget(_ambientCheckBox,0,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Ambient",this),0,1,Qt::AlignHCenter);
    grid->addWidget(_ambientPicker,0,2,Qt::AlignHCenter);

    grid->addWidget(_emissiveCheckBox,1,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Emissive",this),1,1,Qt::AlignHCenter);
    grid->addWidget(_emissivePicker,1,2,Qt::AlignHCenter);

    grid->addWidget(_diffuseCheckBox,2,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Diffuse",this),2,1,Qt::AlignHCenter);
    grid->addWidget(_diffusePicker,2,2,Qt::AlignHCenter);

    grid->addWidget(_specularCheckBox,3,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Specular",this),3,1,Qt::AlignHCenter);
    grid->addWidget(_specularPicker,3,2,Qt::AlignHCenter);

    grid->addWidget(_shininessCheckBox,4,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Shininess",this),4,1,Qt::AlignHCenter);
    grid->addWidget(_shininessEdit,4,2,Qt::AlignHCenter);

    layout->addLayout(hlayout);
    layout->addLayout(grid);


    connect(_nameEdit, SIGNAL( textChanged( const QString & ) ), this , SLOT( setWidgetDirty() ) );
    connect(_shininessEdit, SIGNAL( textChanged( const QString & ) ), this , SLOT( setWidgetDirty() ) );

    connect(_ambientCheckBox, SIGNAL( toggled( bool ) ), this , SLOT( setWidgetDirty() ) );
    connect(_ambientCheckBox, SIGNAL( toggled( bool ) ), _ambientPicker , SLOT( setEnabled(bool ) ) );

    connect(_emissiveCheckBox, SIGNAL( toggled( bool ) ), this , SLOT( setWidgetDirty() ) );
    connect(_emissiveCheckBox, SIGNAL( toggled( bool ) ), _emissivePicker, SLOT( setEnabled(bool) ) );

    connect(_specularCheckBox, SIGNAL( toggled( bool ) ), this , SLOT( setWidgetDirty() ) );
    connect(_specularCheckBox, SIGNAL( toggled( bool ) ), _specularPicker, SLOT( setEnabled(bool) ) );

    connect(_diffuseCheckBox, SIGNAL( toggled( bool ) ), this , SLOT( setWidgetDirty() ) );
    connect(_diffuseCheckBox, SIGNAL( toggled( bool ) ), _diffusePicker, SLOT( setEnabled(bool) ) );

    connect(_shininessCheckBox, SIGNAL( toggled( bool ) ), this, SLOT( setWidgetDirty() ) );
    connect(_shininessCheckBox, SIGNAL( toggled( bool ) ), _shininessEdit, SLOT( setEnabled(bool) ) );

    connect(_ambientPicker, SIGNAL( hasChanged() ), this, SLOT( setWidgetDirty() ) );
    connect(_emissivePicker, SIGNAL( hasChanged() ), this, SLOT( setWidgetDirty() ) );
    connect(_specularPicker, SIGNAL( hasChanged() ), this, SLOT( setWidgetDirty() ) );
    connect(_diffusePicker, SIGNAL( hasChanged() ), this, SLOT( setWidgetDirty() ) );

    readFromData();

    return true;
}
void MaterialDataWidget::setDataReadOnly(bool readOnly)
{
    _nameEdit->setReadOnly(readOnly);
    _nameEdit->setEnabled(!readOnly);
    _ambientPicker->setEnabled(!readOnly);
    _ambientCheckBox->setEnabled(!readOnly);
    _emissivePicker->setEnabled(!readOnly);
    _emissiveCheckBox->setEnabled(!readOnly);
    _specularPicker->setEnabled(!readOnly);
    _specularCheckBox->setEnabled(!readOnly);
    _diffusePicker->setEnabled(!readOnly);
    _diffuseCheckBox->setEnabled(!readOnly);
    _shininessEdit->setReadOnly(readOnly);
    _shininessCheckBox->setEnabled(!readOnly);
}
void MaterialDataWidget::readFromData()
{
    using namespace sofa::core::loader;
    const Material& material = getData()->virtualGetValue();
    _nameEdit->setText( QString( material.name.c_str() ) );
    _ambientCheckBox->setChecked( material.useAmbient );
    _emissiveCheckBox->setChecked( material.useEmissive );
    _diffuseCheckBox->setChecked( material.useDiffuse );
    _specularCheckBox->setChecked( material.useSpecular );
    _shininessCheckBox->setChecked(material.useShininess);
    QString str;
    str.setNum(material.shininess);
    _shininessEdit->setText(str);

    _ambientPicker->setColor( material.ambient );
    _emissivePicker->setColor( material.emissive );
    _specularPicker->setColor( material.specular );
    _diffusePicker->setColor( material.diffuse );

    _ambientPicker->setEnabled( _ambientCheckBox->isChecked() );
    _emissivePicker->setEnabled( _emissiveCheckBox->isChecked() );
    _specularPicker->setEnabled( _specularCheckBox->isChecked() );
    _diffusePicker->setEnabled( _diffuseCheckBox->isChecked() );


}
void MaterialDataWidget::writeToData()
{
    using namespace sofa::core::loader;
    Material* material = getData()->virtualBeginEdit();

    material->name      = _nameEdit->text().toStdString();
    material->ambient   = _ambientPicker->getColor();
    material->diffuse   = _diffusePicker->getColor();
    material->emissive  = _emissivePicker->getColor();
    material->specular  = _specularPicker->getColor();
    material->shininess = _shininessEdit->text().toFloat();
    material->useAmbient = _ambientCheckBox->isChecked();
    material->useDiffuse = _diffuseCheckBox->isChecked();
    material->useShininess = _shininessCheckBox->isChecked();
    material->useEmissive = _emissiveCheckBox->isChecked();
    material->useSpecular = _specularCheckBox->isChecked();


    getData()->virtualEndEdit();

}


bool VectorMaterialDataWidget::createWidgets()
{
    if( getData()->virtualGetValue().empty() )
    {
        return false;
    }
    _comboBox = new QComboBox(this);
    _materialDataWidget = new MaterialDataWidget(this,this->objectName().toStdString().c_str(),&_currentMaterial);
    _materialDataWidget->createWidgets();
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(_comboBox);
    layout->addWidget(_materialDataWidget);

    connect( _comboBox, SIGNAL(activated(int)  ), this, SLOT( changeMaterial( int) ) );
    connect( _materialDataWidget, SIGNAL( WidgetDirty(bool) ) ,this, SLOT( setWidgetDirty(bool) ) );

    readFromData();

    return true;
}

void VectorMaterialDataWidget::setDataReadOnly(bool readOnly)
{
    if (_materialDataWidget)
        _materialDataWidget->setDataReadOnly(readOnly);
}

void VectorMaterialDataWidget::readFromData()
{
    VectorMaterial::const_iterator iter;
    const VectorMaterial& vecMaterial = getData()->virtualGetValue();
    if( vecMaterial.empty() )
    {
        return;
    }
    _comboBox->clear();
    _vectorEditedMaterial.clear();
    std::copy(vecMaterial.begin(), vecMaterial.end(), std::back_inserter(_vectorEditedMaterial) );
    for( iter = _vectorEditedMaterial.begin(); iter != _vectorEditedMaterial.end(); ++iter )
    {
        _comboBox->addItem( QString( (*iter).name.c_str() ) );
    }
    _currentMaterialPos = 0;
    _comboBox->setCurrentIndex(_currentMaterialPos);
    _currentMaterial.setValue(_vectorEditedMaterial[_currentMaterialPos]);
    _materialDataWidget->setData(&_currentMaterial);
    _materialDataWidget->updateWidgetValue();
}

void VectorMaterialDataWidget::changeMaterial( int index )
{
    using namespace sofa::core::loader;

    //Save previous Material
    _materialDataWidget->updateDataValue();
    Material mat(_currentMaterial.virtualGetValue() );
    _vectorEditedMaterial[_currentMaterialPos] = mat;

    //Update current Material
    _currentMaterialPos = index;
    _currentMaterial.setValue(_vectorEditedMaterial[index]);

    //Update Widget
    _materialDataWidget->setData(&_currentMaterial);
    _materialDataWidget->updateWidgetValue();
}

void VectorMaterialDataWidget::writeToData()
{
    using namespace sofa::core::loader;

    _materialDataWidget->updateDataValue();
    Material mat(_currentMaterial.virtualGetValue() );
    _vectorEditedMaterial[_currentMaterialPos] = mat;

    VectorMaterial* vecMaterial = getData()->virtualBeginEdit();
    assert(vecMaterial->size() == _vectorEditedMaterial.size() );
    std::copy(_vectorEditedMaterial.begin(), _vectorEditedMaterial.end(), vecMaterial->begin() );

    getData()->virtualEndEdit();

}

}
}
}
