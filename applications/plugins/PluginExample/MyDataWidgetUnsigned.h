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
#ifndef PLUGINEXAMPLE_MYDATAWIDGETUNSIGNED_H
#define PLUGINEXAMPLE_MYDATAWIDGETUNSIGNED_H

#include <PluginExample/config.h>

#include <sofa/gui/qt/DataWidget.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QSlider>
#include <QString>


namespace sofa
{

namespace gui
{

namespace qt
{


/**
 * \brief Customization of the representation of Data<unsigned> types
 * in the gui. In the .cpp file this widget is registered to represent
 * myData from MyBehaviorModel in the gui.
 **/
class MyDataWidgetUnsigned : public TDataWidget<unsigned>
{
    Q_OBJECT
public :
    // The class constructor takes a TData<unsigned> since it creates
    // a widget for a that particular data type.
    MyDataWidgetUnsigned(QWidget* parent, const char* name, core::objectmodel::Data<unsigned> *data):
        TDataWidget<unsigned>(parent, name,data) {};

    // In this method we  create the widgets and perform the signal / slots
    // connections.
    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);
protected slots:
    void change();
protected:
    ///Implements how update the widgets knowing the data value.
    virtual void readFromData();
    ///Implements how to update the data, knowing the widget value.
    virtual void writeToData();
    QSlider *qslider;
    QLabel *label1;
    QLabel *label2;
};


} // namespace qt

} // namespace gui

} // namespace sofa


#endif // PLUGINEXAMPLE_MYDATAWIDGETUNSIGNED_H
