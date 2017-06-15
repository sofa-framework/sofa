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
#ifndef SOFA_GUI_QT_TRANSFORMATIONWIDGET_H
#define SOFA_GUI_QT_TRANSFORMATIONWIDGET_H

#include <sofa/simulation/Node.h>
#include "WDoubleLineEdit.h"

#include <QWidget>
#include <QTextEdit>
#include <QGroupBox>

namespace sofa
{
namespace gui
{
namespace qt
{

struct ModifyObjectFlags;
class QTransformationWidget : public QGroupBox
{
    Q_OBJECT
public:
    QTransformationWidget(QWidget* parent, QString name);
    unsigned int getNumWidgets() const { return numWidgets_;};

    void setDefaultValues();
    bool isDefaultValues() const;
    void applyTransformation(simulation::Node *node);
public slots:
    void changeValue() {emit TransformationDirty(true);}
signals:
    void TransformationDirty(bool);
protected:
    const unsigned int numWidgets_;

    WDoubleLineEdit* translation[3];
    WDoubleLineEdit* rotation[3];
    WDoubleLineEdit* scale[3];
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_TRANSFORMATIONWIDGET_H

