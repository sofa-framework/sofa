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
#include <SofaGeneralEngine/IndicesFromValues.h>
using sofa::component::engine::IndicesFromValues;

#include <sofa/core/ObjectFactory.h>
using sofa::core::objectmodel::New ;

#include <sofa/defaulttype/RigidTypes.h>
#include <SofaTest/Sofa_test.h>

namespace sofa
{
struct TestIndicesFromValues : public Sofa_test<>{
    // Test computation on a simple example
    void search_one_index(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("values")->read("-1");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(), "7");
    }

    void search_two_indices(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("values")->read("-1. 1.");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(), "7 5");
    }


    void search_nothing(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("values")->read("");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(), "");
    }

    void search_in_nothing(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read(" ");
        m_thisObject->findData("values")->read("1");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(),"");
    }

    void search_nothing_in_nothing(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read(" ");
        m_thisObject->findData("values")->read(" ");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(),"");
    }

    void search_existing_and_nonexisting(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("values")->read("1.  4. ");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(),"5");
    }


    void search_nonexisting(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("values")->read("4. ");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(), "");
    }

    void search_a_sequence(){
        IndicesFromValues<SReal>::SPtr m_thisObject=New<IndicesFromValues<SReal>>();
        m_thisObject->findData("global")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("values")->read("1. 0. -1. ");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(), "");
    }


};

TEST_F(TestIndicesFromValues, search_one_index ) { ASSERT_NO_THROW(this->search_one_index()); }

TEST_F(TestIndicesFromValues, search_two_indices ) { ASSERT_NO_THROW(this->search_two_indices()); }

TEST_F(TestIndicesFromValues, search_nothing ) { ASSERT_NO_THROW(this->search_nothing()); }

TEST_F(TestIndicesFromValues, search_in_nothing ) { ASSERT_NO_THROW(this->search_in_nothing()); }

TEST_F(TestIndicesFromValues, search_nothing_in_nothing ) { ASSERT_NO_THROW(this->search_nothing_in_nothing()); }

TEST_F(TestIndicesFromValues, search_existing_and_nonexisting) { ASSERT_NO_THROW(this->search_existing_and_nonexisting()); }

TEST_F(TestIndicesFromValues, search_nonexisting) { ASSERT_NO_THROW(this->search_nonexisting()); }


} // namespace sofa
