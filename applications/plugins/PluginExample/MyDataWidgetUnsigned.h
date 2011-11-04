/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_MyDataWidgetUnsigned_H
#define SOFA_GUI_QT_MyDataWidgetUnsigned_H

#include "initPlugin.h"
#include <sofa/gui/qt/DataWidget.h>

#ifdef SOFA_QT4
#include <QLabel>
#include <QVBoxLayout>
#include <QSlider>
#include <QString>
#else
#include <qlabel.h>
#include <qlayout.h>
#include <qslider.h>
#include <qstring.h>
#endif



namespace sofa
{
namespace gui
{
namespace qt
{
/**
*\brief Customization of the representation of Data<unsigned> types
* in the gui. In the .cpp file this widget is registered to represent
* myData from MyBehaviorModel in the gui.
**/
class SOFA_MyPluginExample_API MyDataWidgetUnsigned : public TDataWidget<unsigned>
{
    Q_OBJECT
public :
    ///The class constructor takes a TData<unsigned> since it creates
    ///a widget for a that particular data type.
    MyDataWidgetUnsigned(QWidget* parent, const char* name, core::objectmodel::Data<unsigned>* data):
        TDataWidget<unsigned>(parent,name,data) {};
    ///In this method we  create the widgets and perform the signal / slots
    ///connections.
    virtual bool createWidgets();
protected slots:
    void change();
protected:
    ///Implements how update the widgets knowing the data value.
    virtual void readFromData();
    ///Implements how to update the data, knowing the widget value.
    virtual void writeToData();
    QSlider* qslider;
    QLabel*  label1;
    QLabel*  label2;
};

}

}

}
#endif
