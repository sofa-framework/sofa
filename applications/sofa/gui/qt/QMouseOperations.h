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
#ifndef SOFA_GUI_QT_QMOUSEOPERATIONS_H
#define SOFA_GUI_QT_QMOUSEOPERATIONS_H

#include "SofaGUIQt.h"
#include <sofa/gui/MouseOperations.h>
#include "SofaMouseManager.h"

#include <QWidget>
#include <QLineEdit>
#include <QRadioButton>
#include <QSpinBox>
#include <QSlider>
#include <QPushButton>
#include <QCheckBox>
#include <QGroupBox>
#include <QComboBox>

#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{
class DataWidget;

class SOFA_SOFAGUIQT_API QMouseOperation : public QWidget
{
    Q_OBJECT
public:
    DataWidget *createWidgetFromData(sofa::core::objectmodel::BaseData* data);

public slots:
    void WidgetDirty(bool);
};

class QAttachOperation : public QMouseOperation, public AttachOperation
{
    Q_OBJECT
public:
    QAttachOperation();
    void configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button);

protected:
    DataWidget *stiffnessWidget;
    DataWidget *arrowSizeWidget;
    DataWidget *showSizeFactorWidget;
};

class QAddRecordedCameraOperation : public QMouseOperation, public AddRecordedCameraOperation
{
    Q_OBJECT
public:
    QAddRecordedCameraOperation();
    void configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button);
};

class QStartNavigationOperation : public QMouseOperation, public StartNavigationOperation  
{
    Q_OBJECT
public:
    QStartNavigationOperation();
    void configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button);
};

class QFixOperation : public QMouseOperation, public FixOperation
{
    Q_OBJECT
public:
    QFixOperation();
    void configure(PickHandler *picker, sofa::component::configurationsetting::MouseButtonSetting* button);

protected:
    DataWidget *stiffnessWidget;
};




class QInciseOperation : public QWidget, public InciseOperation
{
    Q_OBJECT
public:
    QInciseOperation();
    int getIncisionMethod() const;
    int getSnapingBorderValue() const;
    int getSnapingValue() const;

    bool getCompleteIncision () {return finishIncision;}
    bool getKeepPoint () {return keepPoint;}

    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        InciseOperation::configure(picker, b);
    }

    bool finishIncision;
    bool keepPoint;

public slots:
    void setEnableBox (bool i);
    void setFinishIncision (bool i);
    void setkeepPoint (bool i);


protected:
    QGroupBox* incisionMethodChoiceGroup;
    QRadioButton* method1;
    QRadioButton* method2;

    QGroupBox *advancedOperations;
    QCheckBox *finishCut;
    QCheckBox *storeLastPoint;

    QGroupBox* advancedOptions;
    QSlider  *snapingBorderSlider;
    QSpinBox *snapingBorderValue;
    QSlider  *snapingSlider;
    QSpinBox *snapingValue;
};




class QTopologyOperation : public QWidget, public TopologyOperation
{
    Q_OBJECT
public:
    QTopologyOperation();
    double getScale() const;
    int getTopologicalOperation() const;
    bool getVolumicMesh() const;



    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        TopologyOperation::configure(picker, b);
    }

public slots:
    void setEnableBox (int i);

protected:

    QComboBox *operationChoice;
    QRadioButton *meshType1;
    QRadioButton *meshType2;

    QGroupBox *advancedOptions;
    QSlider *scaleSlider;
    QSpinBox *scaleValue;
};


class QAddSutureOperation : public QWidget, public AddSutureOperation
{
    Q_OBJECT
public:
    QAddSutureOperation();
    double getStiffness() const;
    double getDamping() const;

    void configure(PickHandler *picker, MOUSE_BUTTON b)
    {
        AddSutureOperation::configure(picker, b);
    }

protected:
    QLineEdit *stiffness;
    QLineEdit *damping;
};

}
}
}

#endif
