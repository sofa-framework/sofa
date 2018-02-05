/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaGeneralEngine/IndexValueMapper.h>
using sofa::component::engine::IndexValueMapper;

#include <sofa/core/ObjectFactory.h>
using sofa::core::objectmodel::New ;

#include <SofaTest/Sofa_test.h>

using testing::Types;
using sofa::defaulttype::Vec3fTypes ;
using sofa::defaulttype::Vec3dTypes ;

namespace sofa
{
template <class T>
struct TestIndexValueMapper : public Sofa_test<>
{

    void input_to_output_empty_values()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "");
    }

    void input_to_output_values()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("1.");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "1");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "1");
    }

    void input_to_output_float_values()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("1.5");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "1.5");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "1.5");
    }

    void input_to_output_negative_values()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("-1.5");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "-1.5");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "-1.5");
    }

    void resize()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("1.5");
        m_thisObject->findData("indices")->read("4");
        m_thisObject->findData("value")->read("2");
        m_thisObject->findData("defaultValue")->read("0");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "1.5");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "1.5 0 0 0 2");

    }

    void negative_indices()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("1.5");
        {
            EXPECT_MSG_EMIT(Warning);
            m_thisObject->findData("indices")->read("-4");
            EXPECT_EQ(m_thisObject->findData("indices")->getValueString(), "0");
        }
    }

    void change_indice_value()
    {
        typename IndexValueMapper<T>::SPtr m_thisObject = New<IndexValueMapper<T>>();
        m_thisObject->findData("inputValues")->read("1.5");
        m_thisObject->findData("indices")->read("0");
        m_thisObject->findData("value")->read("2");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "1.5");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "2");

        m_thisObject->findData("inputValues")->read("1.5 2 2.5");
        m_thisObject->findData("indices")->read("2");
        m_thisObject->findData("value")->read("3");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("inputValues")->getValueString(), "1.5 2 2.5");
        EXPECT_EQ(m_thisObject->findData("outputValues")->getValueString(), "1.5 2 3");
    }

};

typedef Types<
Vec3Types
#ifdef SOFA_WITH_DOUBLE
,Vec3dTypes
#endif
#ifdef SOFA_WITH_FLOAT
,Vec3fTypes
#endif
> DataTypes;

TYPED_TEST_CASE(TestIndexValueMapper, DataTypes);

TYPED_TEST(TestIndexValueMapper, input_to_output_empty_values)
{
    ASSERT_NO_THROW(this->input_to_output_empty_values());
}

TYPED_TEST(TestIndexValueMapper, input_to_output_values)
{
    ASSERT_NO_THROW(this->input_to_output_values());
}

TYPED_TEST(TestIndexValueMapper, input_to_output_float_values)
{
    ASSERT_NO_THROW(this->input_to_output_float_values());
}

TYPED_TEST(TestIndexValueMapper, input_to_output_negative_values)
{
    ASSERT_NO_THROW(this->input_to_output_negative_values());
}

TYPED_TEST(TestIndexValueMapper, resize)
{
    ASSERT_NO_THROW(this->resize());
}

TYPED_TEST(TestIndexValueMapper, negative_indices)
{
    ASSERT_NO_THROW(this->negative_indices());
}

TYPED_TEST(TestIndexValueMapper, change_indice_value)
{
    ASSERT_NO_THROW(this->change_indice_value());
}

} // namespace sofa
