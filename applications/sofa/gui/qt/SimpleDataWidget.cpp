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

#include "SimpleDataWidget.h"
#include <sofa/helper/Factory.inl>
#include <sofa/core/objectmodel/Tag.h>
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Creator;
using sofa::helper::fixed_array;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(SimpleDataWidget);

Creator<DataWidgetFactory, SimpleDataWidget<bool> > DWClass_bool("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<char> > DWClass_char("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<unsigned char> > DWClass_uchar("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<int> > DWClass_int("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<unsigned int> > DWClass_uint("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<float> > DWClass_float("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<double> > DWClass_double("default",true);
Creator<DataWidgetFactory, SimpleDataWidget<std::string> > DWClass_string("default",true);



Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,int> > > DWClass_Vec1i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,unsigned int> > > DWClass_Vec1u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,float> > > DWClass_Vec1f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<1,double> > > DWClass_Vec1d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,int> > > DWClass_Vec2i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,unsigned int> > > DWClass_Vec2u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,float> > > DWClass_Vec2f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<2,double> > > DWClass_Vec2d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,int> > > DWClass_Vec3i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,unsigned int> > > DWClass_Vec3u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,float> > > DWClass_Vec3f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<3,double> > > DWClass_Vec3d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,int> > > DWClass_Vec4i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,unsigned int> > > DWClass_Vec4u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,float> > > DWClass_Vec4f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<4,double> > > DWClass_Vec4d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,int> > > DWClass_Vec6i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,unsigned int> > > DWClass_Vec6u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,float> > > DWClass_Vec6f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<6,double> > > DWClass_Vec6d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<8,int> > > DWClass_Vec8i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Vec<8,unsigned int> > > DWClass_Vec8u("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,1> > > DWClass_fixed_array1i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,1> > > DWClass_fixed_array1u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,2> > > DWClass_fixed_array2i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,2> > > DWClass_fixed_array2u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,3> > > DWClass_fixed_array3i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,3> > > DWClass_fixed_array3u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,4> > > DWClass_fixed_array4i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,4> > > DWClass_fixed_array4u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,6> > > DWClass_fixed_array6i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,6> > > DWClass_fixed_array6u("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<int,8> > > DWClass_fixed_array8i("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<unsigned int,8> > > DWClass_fixed_array8u("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::topology::Topology::Edge        > > DWClass_Edge       ("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::topology::Topology::Triangle    > > DWClass_Triangle   ("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::topology::Topology::Quad        > > DWClass_Quad       ("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::topology::Topology::Tetrahedron > > DWClass_Tetrahedron("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::topology::Topology::Hexahedron  > > DWClass_Hexahedron ("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<float,1> > > DWClass_fixed_array1f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<double,1> > > DWClass_fixed_array1d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<float,2> > > DWClass_fixed_array2f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<double,2> > > DWClass_fixed_array2d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<float,3> > > DWClass_fixed_array3f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<double,3> > > DWClass_fixed_array3d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<float,4> > > DWClass_fixed_array4f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<double,4> > > DWClass_fixed_array4d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<float,6> > > DWClass_fixed_array6f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<double,6> > > DWClass_fixed_array6d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<float,8> > > DWClass_fixed_array8f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< fixed_array<double,8> > > DWClass_fixed_array8d("default",true);

Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Quater<float> > > DWClass_Quatf("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Quater<double> > > DWClass_Quatd("default",true);


using sofa::helper::Polynomial_LD;
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,5> > >DWClass_PolynomialLD5d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,4> > >DWClass_PolynomialLD4d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,3> > >DWClass_PolynomialLD3d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,2> > >DWClass_PolynomialLD2d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<double,1> > >DWClass_PolynomialLD1d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<float ,5> > >DWClass_PolynomialLD5f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<float ,4> > >DWClass_PolynomialLD4f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<float ,3> > >DWClass_PolynomialLD3f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<float ,2> > >DWClass_PolynomialLD2f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::helper::Polynomial_LD<float ,1> > >DWClass_PolynomialLD1f("default",true);
#ifdef TODOLINK
using sofa::core::objectmodel::ObjectRef;
Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::objectmodel::ObjectRef > >DWClass_ObjectRef("default",true);
#endif

Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,2,float> > > DWClass_Mat22f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,2,double> > > DWClass_Mat22d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,3,float> > > DWClass_Mat23f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<2,3,double> > > DWClass_Mat23d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,3,float> > > DWClass_Mat33f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,3,double> > > DWClass_Mat33d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,4,float> > > DWClass_Mat34f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<3,4,double> > > DWClass_Mat34d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<4,4,float> > > DWClass_Mat44f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<4,4,double> > > DWClass_Mat44d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<6,6,float> > > DWClass_Mat66f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< Mat<6,6,double> > > DWClass_Mat66d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::objectmodel::TagSet > > DWClass_TagSet("default",true);


////////////////////////////////////////////////////////////////
/// OptionsGroup support
////////////////////////////////////////////////////////////////

//these functions must be written here for effect of writeToData
Creator<DataWidgetFactory,RadioDataWidget> DWClass_OptionsGroup("default",true);

bool RadioDataWidget::createWidgets()
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    sofa::helper::OptionsGroup m_radiotrick = getData()->virtualGetValue();
    const unsigned int LIMIT_NUM_BUTTON=4;
    buttonMode=m_radiotrick.size() < LIMIT_NUM_BUTTON;
    if (buttonMode)
    {
        buttonList=new QButtonGroup(this);

        for(unsigned int i=0; i<m_radiotrick.size(); i++)
        {
            std::string m_itemstring=m_radiotrick[i];

            QRadioButton * m_radiobutton=new QRadioButton(QString(m_itemstring.c_str()), this);
            if (i==m_radiotrick.getSelectedId()) m_radiobutton->setChecked(true);
            layout->addWidget(m_radiobutton);
            buttonList->addButton(m_radiobutton,i);
        }
        connect(buttonList, SIGNAL(buttonClicked(int)), this, SLOT(setWidgetDirty())) ;
    }
    else
    {
        comboList=new QComboBox(this);
		comboList->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

        sofa::helper::OptionsGroup m_radiotrick = getData()->virtualGetValue();
        QStringList list;
        for(unsigned int i=0; i<m_radiotrick.size(); i++) list << m_radiotrick[i].c_str();

        comboList->insertItems(0, list);

        comboList->setCurrentIndex(m_radiotrick.getSelectedId());

        connect(comboList, SIGNAL(activated(int)), this, SLOT(setWidgetDirty()));
        layout->addWidget(comboList);

    }

    return true;
}
void RadioDataWidget::setDataReadOnly(bool readOnly)
{
    if (buttonMode)
    {
        QList<QAbstractButton *> buttons = buttonList->buttons();
        for (int i = 0; i < buttons.size(); ++i)
        {
            buttons.at(i)->setEnabled(!readOnly);
        }
    }
    else
    {
        comboList->setEnabled(!readOnly);
    }
}

void RadioDataWidget::readFromData()
{
    sofa::helper::OptionsGroup m_radiotrick = getData()->virtualGetValue();

    if (buttonMode)
    {
        buttonList->button(m_radiotrick.getSelectedId())->setChecked(true);
    }
    else
    {
        comboList->setCurrentIndex(m_radiotrick.getSelectedId());
    }
}
void RadioDataWidget::writeToData()
{
    sofa::helper::OptionsGroup m_radiotrick = getData()->virtualGetValue();
    if (buttonMode)
    {
        m_radiotrick.setSelectedItem((unsigned int)buttonList->checkedId ());
    }
    else
    {
        m_radiotrick.setSelectedItem((unsigned int)comboList->currentIndex());
    }

    this->getData()->virtualSetValue(m_radiotrick);
}



} // namespace qt

} // namespace gui

} // namespace sofa
