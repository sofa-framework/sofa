/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/core/IntrusiveObject.h>
#include <gtest/gtest.h>
#include <sofa/core/sptr.h>


class DummyIntrusiveObject : public sofa::core::IntrusiveObject
{
public:
    DummyIntrusiveObject() = default;
    explicit DummyIntrusiveObject(const std::function<void()>& _destructorCallback)
        : destructorCallback(_destructorCallback) {}

    ~DummyIntrusiveObject() override
    {
        destructorCallback();
    }

private:
    std::function<void()> destructorCallback;
};



TEST(IntrusiveObject, IsDestructorCalled)
{
    std::size_t nbTimesDestructorCalled = 0;
    {
        sofa::core::sptr<DummyIntrusiveObject> dummy(new DummyIntrusiveObject([&nbTimesDestructorCalled]()
        {
            nbTimesDestructorCalled++;
        }));
    }
    EXPECT_EQ(1, nbTimesDestructorCalled);
}


TEST(IntrusiveObject, Copy)
{
    std::size_t nbTimesDestructorCalled = 0;
    {
        sofa::core::sptr<DummyIntrusiveObject> dummy0;
        {
            sofa::core::sptr<DummyIntrusiveObject> dummy(new DummyIntrusiveObject([&nbTimesDestructorCalled]()
            {
                nbTimesDestructorCalled++;
            }));

            dummy0 = dummy;
        }
    }
    EXPECT_EQ(1, nbTimesDestructorCalled);
}
