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
#ifndef PLUGINS_PIM_GUI_QT_QMOUSEOPERATIONS_H
#define PLUGINS_PIM_GUI_QT_QMOUSEOPERATIONS_H

#include <PhysicsBasedInteractiveModeler/gui/MouseOperations.h>
//#include <sofa/gui/qt/SofaMouseManager.h>
#include <sofa/gui/PickHandler.h>
#ifdef SOFA_QT4
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QLineEdit>
#include <QSpinBox>
#include <QSlider>
#include <QRadioButton>
#include <QPushButton>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#else
#include <qwidget.h>
#include <qlineedit.h>
#include <qspinbox.h>
#include <qslider.h>
#include <qradiobutton.h>
#include <qpushbutton.h>
#include <qgroupbox>
#include <qlabel.h>
#include <qlineedit.h>
#endif
#include <iostream>

namespace plugins
{
namespace pim
{
namespace gui
{
namespace qt
{

using namespace sofa::gui;

class QSculptOperation : public QWidget, public SculptOperation
{
    Q_OBJECT
public:
    QSculptOperation();
    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        SculptOperation::configure(picker, b);
    }

    double getForce() const;
    double getScale() const;
    double getMass() const;
    double getStiffness() const;
    double getDamping() const;
    bool isCheckedFix() const;
    bool isCheckedInflate() const;
    bool isCheckedDeflate() const;

public slots:
    void setScale();
    void animate();
    void updateInterface(bool);

protected:

    QGroupBox *options;

    QSlider  *forceSlider;
    QSpinBox *forceValue;

    QSlider  *scaleSlider;
    QSpinBox *scaleValue;

    QLineEdit *massValue;
    QLineEdit *stiffnessValue;
    QLineEdit *dampingValue;

    QRadioButton *inflateRadioButton;
    QRadioButton *deflateRadioButton;
    QRadioButton *fixRadioButton;

    QPushButton *animatePushButton;

    QHBoxLayout *HLayout1;
    QHBoxLayout *HLayout2;
    QHBoxLayout *HLayout3;
    QVBoxLayout *VLayout;

    QLabel *forceLabel;
    QLabel *massLabel;
    QLabel *stiffnessLabel;
    QLabel *dampingLabel;
};

} // namespace qt
} // namespace gui
} // namespace pim
} // namespace plugins

#endif
