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

/// Private namespace
namespace sofa::gui::qt::qrgbacolorpicker_h
{

using sofa::type::Vec4f ;

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
    void setColor(const type::RGBAColor& color);

    void setColor( const Vec4f& color );
    type::RGBAColor getColor() const;

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

} // namespace sofa::gui::qt::qrgbacolorpicker_h

namespace sofa::gui::qt
{
    using sofa::gui::qt::qrgbacolorpicker_h::QRGBAColorPicker;
} // namespace sofa::gui::qt
