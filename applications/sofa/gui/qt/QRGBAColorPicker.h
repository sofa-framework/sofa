/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef QRGBACOLORPICKER_H
#define QRGBACOLORPICKER_H
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
/// Private namespace
namespace qrgbacolorpicker_h
{
using sofa::defaulttype::Vec4f ;

/**
    @class QRGBAColorPicker
    Implement a widget to select a color either using a color wheel or by three values.
*/
class QRGBAColorPicker : public QWidget
{
    Q_OBJECT

signals:
    void hasChanged();

public:
    QRGBAColorPicker(QWidget* parent);
    void setColor( const Vec4f& color );
    Vec4f getColor() const;

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

} /// namespace qrgbacolorpicker_h

using qrgbacolorpicker_h::QRGBAColorPicker ;

} /// namespace qt
} /// namespace gui
} /// namespace sofa

#endif //QRGBACOLORPICKER

