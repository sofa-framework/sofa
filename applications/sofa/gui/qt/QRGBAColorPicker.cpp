/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <limits>
#include "QRGBAColorPicker.h"

namespace sofa
{
namespace gui
{
namespace qt
{
namespace qrgbacolorpicker_h
{

QRGBAColorPicker::QRGBAColorPicker(QWidget* parent) : QWidget(parent)
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

Vec4f QRGBAColorPicker::getColor() const
{
    typedef unsigned char uchar;
    const uchar max = std::numeric_limits<uchar>::max();
    Vec4f color;
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

void QRGBAColorPicker::updateRGBAColor()
{
    typedef unsigned char uchar;
    Vec4f color;
    const uchar r = _r->text().toInt();
    const uchar g = _g->text().toInt();
    const uchar b = _b->text().toInt();
    const uchar a = _a->text().toInt();
    _rgba = qRgba(r,g,b,a);
    redrawColorButton();
}

void QRGBAColorPicker::setColor(const Vec4f& color)
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

void QRGBAColorPicker::redrawColorButton()
{
    int w=_colorButton->width();
    int h=_colorButton->height();

    QPixmap *pix=new QPixmap(25,20);
    pix->fill(QColor(qRed(_rgba),
                     qGreen(_rgba),
                     qBlue(_rgba)));
    _colorButton->setIcon(QIcon(*pix));

    _colorButton->resize(w,h);
}

void QRGBAColorPicker::raiseQColorDialog()
{
    typedef unsigned char uchar;
    const uchar max = std::numeric_limits<uchar>::max();
    int r,g,b,a;
    bool ok;

    Vec4f color;
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

} // namespace q
} // qt
} // gui
} // sofa
