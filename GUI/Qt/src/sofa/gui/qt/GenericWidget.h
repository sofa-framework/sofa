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

#include <sofa/gui/qt/DataWidget.h>

namespace sofa::gui::qt
{
template <class DATA, class WIDGET>
class GenericDataWidget : public sofa::gui::qt::DataWidget
{
public:
    typedef DATA MyData;
    typedef WIDGET MyWidget;

    GenericDataWidget(QWidget* parent, const char* name, MyData* d) : sofa::gui::qt::DataWidget(parent, name, d), m_data(d) {}

    virtual bool createWidgets()
    {
        m_widget = new MyWidget(this, *m_data);
        m_widget->setParent(this);
        setLayout(new QVBoxLayout(this));
        layout()->addWidget(m_widget);
        m_widget->setVisible(true);
        readFromData();
        return true;
    }

    template <class RealObject>
    static RealObject* create(RealObject*, CreatorArgument& arg)
    {
        typename RealObject::MyData* realData = dynamic_cast<typename RealObject::MyData*>(arg.data);
        if (!realData)
            return nullptr;
        else
        {
            RealObject* obj = new RealObject(arg.parent, arg.name.c_str(), realData);
            if (!obj->createWidgets())
            {
                delete obj;
                obj = nullptr;
            }
            if (obj)
            {
                obj->setDataReadOnly(arg.readOnly);
            }
            return obj;
        }
    }

    virtual void setDataReadOnly(bool readOnly) { m_widget->setEnabled(!readOnly); }

    virtual void readFromData() { m_widget->readFromData(*m_data); }

    virtual void writeToData() { m_widget->writeToData(*m_data); }

protected:
    MyData* m_data;
    MyWidget* m_widget;
};

}  // namespace sofa::gui::qt

template <class DATA, class WIDGET>
using GenericDataWidget = sofa::gui::qt::GenericDataWidget<DATA, WIDGET>;
