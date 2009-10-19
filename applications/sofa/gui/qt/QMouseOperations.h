/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_QMOUSEOPERATIONS_H
#define SOFA_GUI_QT_QMOUSEOPERATIONS_H

#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/qt/SofaMouseManager.h>
#ifdef SOFA_QT4
#include <QWidget>
#include <QLineEdit>
#include <QSpinBox>
#include <QSlider>
#include <QRadioButton>
#include <QPushButton>
#else
#include <qwidget.h>
#include <qlineedit.h>
#include <qspinbox.h>
#include <qslider.h>
#include <qradiobutton.h>
#include <qpushbutton.h>
#endif
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{


class QAttachOperation : public QWidget, public AttachOperation
{
    Q_OBJECT
public:
    QAttachOperation();

    double getStiffness() const;
    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        AttachOperation::configure(picker, b);
    }
protected:
    QLineEdit *value;
};



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
    bool isCheckedFix() const;

public slots:
    void setScale();
    void animate(bool checked);

protected:
    QSlider  *forceSlider;
    QSpinBox *forceValue;

    QSlider  *scaleSlider;
    QSpinBox *scaleValue;

    QRadioButton *sculptRadioButton;
    QRadioButton *fixRadioButton;

    QPushButton *animatePushButton;
};



class QFixOperation : public QWidget, public FixOperation
{
    Q_OBJECT
public:
    QFixOperation();
    double getStiffness() const;
    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        FixOperation::configure(picker, b);
    }

protected:
    QLineEdit *value;
};


class QInjectOperation : public QWidget, public InjectOperation
{
    Q_OBJECT
public:
    QInjectOperation();
    double getPotentialValue() const;
    std::string getStateTag() const;
    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        InjectOperation::configure(picker, b);
    }

protected:
    QLineEdit *value;
    QLineEdit *tag;
};

}
}
}

#endif
