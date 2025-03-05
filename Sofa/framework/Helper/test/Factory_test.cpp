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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/helper/Factory.inl>
#include <functional>

namespace sofa
{

/// Enum class used as the key of a factory
enum class DummyEnum
{
    A, B, C, D
};

/// Factories require this operator in order to have a custom class as a key
std::ostream& operator << ( std::ostream& out, const DummyEnum& d )
{
    switch (d)
    {
        case DummyEnum::A: out << "A";
        case DummyEnum::B: out << "B";
        case DummyEnum::C: out << "C";
        case DummyEnum::D: out << "D";
    }
    return out;
}

struct DummyBaseClass
{
    virtual std::string getTestValue() const { return "Base";}

    /// Helper function required by the factory to instantiate a new object
    template<class T>
    static T* create(T*, sofa::helper::NoArgument /*noarg*/)
    {
        return new T();
    }
};
struct DummyClassA : public DummyBaseClass { std::string getTestValue() const override { return "A";}};
struct DummyClassB : public DummyBaseClass { std::string getTestValue() const override { return "B";}};
struct DummyClassC : public DummyBaseClass { std::string getTestValue() const override { return "C";}};
struct DummyClassD : public DummyBaseClass { std::string getTestValue() const override { return "D";}};

/// Definition of the factory
/// Key is the enum type defined earlier. It requires the less operator and operator <<
/// Objects created based on the key are of type DummyBaseClass
using DummyEnumFactory = sofa::helper::Factory<DummyEnum, DummyBaseClass>;

namespace helper
{
template class
sofa::helper::Factory< DummyEnum, DummyBaseClass>;
}

class Factory_test : public BaseTest
{
public:
    void testEnumKey()
    {
        sofa::helper::Creator<DummyEnumFactory, DummyClassA> dummyClassACreator(DummyEnum::A, false);
        sofa::helper::Creator<DummyEnumFactory, DummyClassB> dummyClassBCreator(DummyEnum::B, false);
        sofa::helper::Creator<DummyEnumFactory, DummyClassC> dummyClassCCreator(DummyEnum::C, false);
        sofa::helper::Creator<DummyEnumFactory, DummyClassD> dummyClassDCreator(DummyEnum::D, false);

        const auto a = DummyEnumFactory::CreateObject(DummyEnum::A, sofa::helper::NoArgument());
        EXPECT_TRUE(a);
        EXPECT_TRUE(dynamic_cast<DummyClassA*>(a));
        EXPECT_EQ(a->getTestValue(), "A");

        const auto b = DummyEnumFactory::CreateObject(DummyEnum::B, sofa::helper::NoArgument());
        EXPECT_TRUE(b);
        EXPECT_TRUE(dynamic_cast<DummyClassB*>(b));
        EXPECT_EQ(b->getTestValue(), "B");

        const auto c = DummyEnumFactory::CreateObject(DummyEnum::C, sofa::helper::NoArgument());
        EXPECT_TRUE(c);
        EXPECT_TRUE(dynamic_cast<DummyClassC*>(c));
        EXPECT_EQ(c->getTestValue(), "C");

        const auto d = DummyEnumFactory::CreateObject(DummyEnum::D, sofa::helper::NoArgument());
        EXPECT_TRUE(d);
        EXPECT_TRUE(dynamic_cast<DummyClassD*>(d));
        EXPECT_EQ(d->getTestValue(), "D");

        DummyEnumFactory::ResetEntry(DummyEnum::A);
        DummyEnumFactory::ResetEntry(DummyEnum::B);
        DummyEnumFactory::ResetEntry(DummyEnum::C);
        DummyEnumFactory::ResetEntry(DummyEnum::D);

    }
};

TEST_F(Factory_test, EnumKey)
{
    testEnumKey();
}

} //namespace sofa