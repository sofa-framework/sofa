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
#include <sofa/gui/qt/config.h>
#include <sofa/gui/common/MouseOperations.h>
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

namespace sofa::gui::qt
{
class DataWidget;

class SOFA_GUI_QT_API QMouseOperation : public QWidget
{
    Q_OBJECT
public:
    DataWidget *createWidgetFromData(sofa::core::objectmodel::BaseData* data);

public slots:
    void WidgetDirty(bool);
};

class QAttachOperation : public QMouseOperation, public common::AttachOperation
{
    Q_OBJECT
public:
    QAttachOperation();
    void configure(common::PickHandler *picker, sofa::component::setting::MouseButtonSetting* button) override;

protected:
    DataWidget *stiffnessWidget;
    DataWidget *arrowSizeWidget;
    DataWidget *showSizeFactorWidget;
};

class QAddRecordedCameraOperation : public QMouseOperation, public common::AddRecordedCameraOperation
{
    Q_OBJECT
public:
    QAddRecordedCameraOperation();
    void configure(common::PickHandler *picker, sofa::component::setting::MouseButtonSetting* button) override;
};

class QStartNavigationOperation : public QMouseOperation, public common::StartNavigationOperation
{
    Q_OBJECT
public:
    QStartNavigationOperation();
    void configure(common::PickHandler *picker, sofa::component::setting::MouseButtonSetting* button) override;
};

class QFixOperation : public QMouseOperation, public common::FixOperation
{
    Q_OBJECT
public:
    QFixOperation();
    void configure(common::PickHandler *picker, sofa::component::setting::MouseButtonSetting* button) override;

protected:
    DataWidget *stiffnessWidget;
};




class QInciseOperation : public QWidget, public common::InciseOperation
{
    Q_OBJECT
public:
    QInciseOperation();
    int getIncisionMethod() const override;
    int getSnapingBorderValue() const override;
    int getSnapingValue() const override;

    bool getCompleteIncision () override {return finishIncision;}
    bool getKeepPoint () override {return keepPoint;}

    void configure(common::PickHandler *picker, common::MOUSE_BUTTON b) override
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




class QTopologyOperation : public QWidget, public common::TopologyOperation
{
    Q_OBJECT
public:
    QTopologyOperation();
    double getScale() const override;
    int getTopologicalOperation() const override;
    bool getVolumicMesh() const override;



    void configure(common::PickHandler *picker, common::MOUSE_BUTTON b) override
    {
        common::TopologyOperation::configure(picker, b);
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


class QAddSutureOperation : public QWidget, public common::AddSutureOperation
{
    Q_OBJECT
public:
    QAddSutureOperation();
    double getStiffness() const override;
    double getDamping() const override;

    void configure(common::PickHandler *picker, common::MOUSE_BUTTON b) override
    {
        common::AddSutureOperation::configure(picker, b);
    }

protected:
    QLineEdit *stiffness;
    QLineEdit *damping;
};

} // namespace sofa::gui::qt
