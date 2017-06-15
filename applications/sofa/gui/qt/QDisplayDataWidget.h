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
#ifndef SOFA_GUI_QT_DISPLAYDATAWIDGET_H
#define SOFA_GUI_QT_DISPLAYDATAWIDGET_H

#include "DataWidget.h"
#include "ModifyObject.h"

#include <QWidget>
#include <QLineEdit>
#include <QTextEdit>
#include <QGroupBox>
#include <QSlider>

namespace sofa
{
namespace core
{
namespace objectmodel
{
class BaseData;
}
}
namespace gui
{
namespace qt
{

class DataWidget;
class QDisplayDataInfoWidget;
struct ModifyObjectFlags;

class QDisplayDataWidget : public QGroupBox
{
    Q_OBJECT
public:
    QDisplayDataWidget(QWidget* parent,
            core::objectmodel::BaseData* data,
            const ModifyObjectFlags& flags);
    unsigned int getNumWidgets() const { return numWidgets_;}

    ModifyObjectFlags flag() {return flags_;}

public slots:
    void UpdateData();              //QWidgets ---> BaseData
    void UpdateWidgets();           //BaseData ---> QWidget
signals:
    void WidgetDirty(bool);
    void WidgetUpdate();
    void DataUpdate();
    void DataOwnerDirty(bool);
    void dataValueChanged(QString);

protected:
	static QIcon& RefreshIcon()
	{
		static QIcon icon;
		if(icon.isNull())
		{
			std::string filename = "textures/refresh.png";
			sofa::helper::system::DataRepository.findFile(filename);
			icon = QIcon(filename.c_str());
		}
		return icon;
	}

protected:
    core::objectmodel::BaseData* data_;
    ModifyObjectFlags flags_;
    QDisplayDataInfoWidget*  datainfowidget_;
    DataWidget* datawidget_;
    unsigned int numWidgets_;
    QGridLayout* gridLayout_;
};



class QDataSimpleEdit : public DataWidget
{
    Q_OBJECT
    typedef enum QEditType { TEXTEDIT, LINEEDIT } QEditType;
    typedef union QEditWidgetPtr
    {
        QLineEdit* lineEdit;
        QTextEdit* textEdit;
    } QEditWidgetPtr;

    typedef struct QSimpleEdit
    {
        QEditType type;
        QEditWidgetPtr widget;
    } QSimpleEdit;
public :
    QDataSimpleEdit(QWidget*, const char* name, core::objectmodel::BaseData*);
    virtual unsigned int numColumnWidget() {return 3;}
    virtual unsigned int sizeWidget() {return 6;}
    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);
protected:
    virtual void readFromData();
    virtual void writeToData();
    QSimpleEdit innerWidget_;
};

class QPoissonRatioWidget : public TDataWidget<double>
{
    Q_OBJECT
public :
    QPoissonRatioWidget(QWidget*, const char*, core::objectmodel::Data<double>*);
    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);

protected slots :
    void changeLineEditValue();
    void changeSliderValue();

protected:
    virtual void readFromData();
    virtual void writeToData();
    QSlider* slider;
    QLineEdit* lineEdit;

};





} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_DISPLAYDATAWIDGET_H

