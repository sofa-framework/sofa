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
#include <unordered_set>
#include <sofa/helper/NameDecoder.h>
#include <sofa/testing/BaseTest.h>
#include <sofa/type/fixed_array.h>

#include <sofa/type/Mat.h>

class SimpleClass {};
class _UnderscoreClass {};

template<class T>
class TemplateClass {};

class OuterSimpleClass
{
public:
    class InnerSimpleClass {};
    class InnerSimpleStruct {};
};

namespace sofa::__sofa__
{
    class SimpleClass {};

    class OuterSimpleClass
    {
    public:
        class InnerSimpleClass {};
        class InnerSimpleStruct {};
    };

    template<class T>
    class OuterTemplateClass
    {
    public:
        template<class U>
        class Inner_Template_Class {};
    };
}

template<class T> std::string getDecodeFullName()
{
    return sofa::helper::NameDecoder::decodeFullName(typeid(T));
}
template<class T> std::string getDecodeTypeName()
{
    return sofa::helper::NameDecoder::decodeTypeName(typeid(T));
}
template<class T> std::string getDecodeClassName()
{
    return sofa::helper::NameDecoder::decodeClassName(typeid(T));
}
template<class T> std::string getDecodeNamespaceName()
{
    return sofa::helper::NameDecoder::decodeNamespaceName(typeid(T));
}
template<class T> std::string getDecodeTemplateName()
{
    return sofa::helper::NameDecoder::decodeTemplateName(typeid(T));
}

TEST(NameDecoder_test, simpleClass)
{
    EXPECT_NE(getDecodeFullName<SimpleClass>().find("SimpleClass"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<SimpleClass>(), "SimpleClass");
    EXPECT_EQ(getDecodeClassName<SimpleClass>(), "SimpleClass");
    EXPECT_EQ(getDecodeNamespaceName<SimpleClass>(), "");
    EXPECT_EQ(getDecodeTemplateName<SimpleClass>(), "");
}

TEST(NameDecoder_test, namespaceSimpleClass)
{
    EXPECT_NE(getDecodeFullName<sofa::__sofa__::SimpleClass>().find("sofa::__sofa__::SimpleClass"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<sofa::__sofa__::SimpleClass>(), "SimpleClass");
    EXPECT_EQ(getDecodeClassName<sofa::__sofa__::SimpleClass>(), "SimpleClass");
    EXPECT_EQ(getDecodeNamespaceName<sofa::__sofa__::SimpleClass>(), "sofa::__sofa__");
    EXPECT_EQ(getDecodeTemplateName<sofa::__sofa__::SimpleClass>(), "");
}

TEST(NameDecoder_test, templateClass)
{
    //template type composed of two words: 'unsigned' and 'int'
    EXPECT_NE(getDecodeFullName<TemplateClass<unsigned int> >().find("TemplateClass<unsigned int>"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<TemplateClass<unsigned int> >(), "TemplateClass<unsigned int>");
    EXPECT_EQ(getDecodeClassName<TemplateClass<unsigned int> >(), "TemplateClass");
    EXPECT_EQ(getDecodeNamespaceName<TemplateClass<unsigned int> >(), "");
    EXPECT_EQ(getDecodeTemplateName<TemplateClass<unsigned int> >(), "unsigned int");

    //template is an alias
    using B = double;
    EXPECT_NE(getDecodeFullName<TemplateClass<B> >().find("TemplateClass<double>"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<TemplateClass<B> >(), "TemplateClass<double>");
    EXPECT_EQ(getDecodeClassName<TemplateClass<B> >(), "TemplateClass");
    EXPECT_EQ(getDecodeNamespaceName<TemplateClass<B> >(), "");
    EXPECT_EQ(getDecodeTemplateName<TemplateClass<B> >(), "double");

    //template type itself templated, including a default parameter
    EXPECT_EQ(getDecodeTypeName<TemplateClass<sofa::type::vector<int> > >(), "TemplateClass<vector<int,CPUMemoryManager<int>>>");
    EXPECT_EQ(getDecodeClassName<TemplateClass<sofa::type::vector<int> > >(), "TemplateClass");
    EXPECT_EQ(getDecodeNamespaceName<TemplateClass<sofa::type::vector<int> > >(), "");
    EXPECT_EQ(getDecodeTemplateName<TemplateClass<sofa::type::vector<int> > >(), "vector<int,CPUMemoryManager<int>>");

    //template type itself templated with 2 template parameters, including one composed of two words. Template parameter is an alias
    using D = sofa::type::fixed_array<unsigned int, 4>;
    EXPECT_NE(getDecodeTypeName<TemplateClass<D> >().find("TemplateClass<fixed_array<unsigned int"), std::string::npos);
    EXPECT_EQ(getDecodeClassName<TemplateClass<D> >(), "TemplateClass");
    EXPECT_EQ(getDecodeNamespaceName<TemplateClass<D> >(), "");
    EXPECT_NE(getDecodeTemplateName<TemplateClass<D> >().find("fixed_array<unsigned int,4"), std::string::npos);

    using E = _UnderscoreClass;
    EXPECT_EQ(getDecodeTypeName<TemplateClass<E> >(), "TemplateClass<_UnderscoreClass>");
    EXPECT_EQ(getDecodeClassName<TemplateClass<E> >(), "TemplateClass");
    EXPECT_EQ(getDecodeNamespaceName<TemplateClass<E> >(), "");
    EXPECT_EQ(getDecodeTemplateName<TemplateClass<E> >(), "_UnderscoreClass");
}

TEST(NameDecoder_test, nestedSimpleClass)
{
    EXPECT_NE(getDecodeFullName<OuterSimpleClass>().find("OuterSimpleClass"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<OuterSimpleClass>(), "OuterSimpleClass");
    EXPECT_EQ(getDecodeClassName<OuterSimpleClass>(), "OuterSimpleClass");
    EXPECT_EQ(getDecodeNamespaceName<OuterSimpleClass>(), "");
    EXPECT_EQ(getDecodeTemplateName<OuterSimpleClass>(), "");

    EXPECT_NE(getDecodeFullName<OuterSimpleClass::InnerSimpleClass>().find("OuterSimpleClass::InnerSimpleClass"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<OuterSimpleClass::InnerSimpleClass>(), "InnerSimpleClass");
    EXPECT_EQ(getDecodeClassName<OuterSimpleClass::InnerSimpleClass>(), "InnerSimpleClass");
    EXPECT_EQ(getDecodeNamespaceName<OuterSimpleClass::InnerSimpleClass>(), "OuterSimpleClass");
    EXPECT_EQ(getDecodeTemplateName<OuterSimpleClass::InnerSimpleClass>(), "");

    EXPECT_NE(getDecodeFullName<OuterSimpleClass::InnerSimpleStruct>().find("OuterSimpleClass::InnerSimpleStruct"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<OuterSimpleClass::InnerSimpleStruct>(), "InnerSimpleStruct");
    EXPECT_EQ(getDecodeClassName<OuterSimpleClass::InnerSimpleStruct>(), "InnerSimpleStruct");
    EXPECT_EQ(getDecodeNamespaceName<OuterSimpleClass::InnerSimpleStruct>(), "OuterSimpleClass");
    EXPECT_EQ(getDecodeTemplateName<OuterSimpleClass::InnerSimpleStruct>(), "");
}

TEST(NameDecoder_test, namespaceNestedSimpleClass)
{
    EXPECT_NE(getDecodeFullName<sofa::__sofa__::OuterSimpleClass>().find("sofa::__sofa__::OuterSimpleClass"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<sofa::__sofa__::OuterSimpleClass>(), "OuterSimpleClass");
    EXPECT_EQ(getDecodeClassName<sofa::__sofa__::OuterSimpleClass>(), "OuterSimpleClass");
    EXPECT_EQ(getDecodeNamespaceName<sofa::__sofa__::OuterSimpleClass>(), "sofa::__sofa__");
    EXPECT_EQ(getDecodeTemplateName<sofa::__sofa__::OuterSimpleClass>(), "");

    EXPECT_NE(getDecodeFullName<sofa::__sofa__::OuterSimpleClass::InnerSimpleClass>().find("sofa::__sofa__::OuterSimpleClass::InnerSimpleClass"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<sofa::__sofa__::OuterSimpleClass::InnerSimpleClass>(), "InnerSimpleClass");
    EXPECT_EQ(getDecodeClassName<sofa::__sofa__::OuterSimpleClass::InnerSimpleClass>(), "InnerSimpleClass");
    EXPECT_EQ(getDecodeNamespaceName<sofa::__sofa__::OuterSimpleClass::InnerSimpleClass>(), "sofa::__sofa__::OuterSimpleClass");
    EXPECT_EQ(getDecodeTemplateName<sofa::__sofa__::OuterSimpleClass::InnerSimpleClass>(), "");

    EXPECT_NE(getDecodeFullName<sofa::__sofa__::OuterSimpleClass::InnerSimpleStruct>().find("sofa::__sofa__::OuterSimpleClass::InnerSimpleStruct"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<sofa::__sofa__::OuterSimpleClass::InnerSimpleStruct>(), "InnerSimpleStruct");
    EXPECT_EQ(getDecodeClassName<sofa::__sofa__::OuterSimpleClass::InnerSimpleStruct>(), "InnerSimpleStruct");
    EXPECT_EQ(getDecodeNamespaceName<sofa::__sofa__::OuterSimpleClass::InnerSimpleStruct>(), "sofa::__sofa__::OuterSimpleClass");
    EXPECT_EQ(getDecodeTemplateName<sofa::__sofa__::OuterSimpleClass::InnerSimpleStruct>(), "");
}

TEST(NameDecoder_test, namespaceNestedTemplateClass)
{
    using A = sofa::__sofa__::OuterTemplateClass<unsigned int>;
    EXPECT_NE(getDecodeFullName<A>().find("sofa::__sofa__::OuterTemplateClass<unsigned int>"), std::string::npos);
    EXPECT_EQ(getDecodeTypeName<A>(), "OuterTemplateClass<unsigned int>");
    EXPECT_EQ(getDecodeClassName<A>(), "OuterTemplateClass");
    EXPECT_EQ(getDecodeNamespaceName<A>(), "sofa::__sofa__");
    EXPECT_EQ(getDecodeTemplateName<A>(), "unsigned int");

    using B = sofa::__sofa__::OuterTemplateClass<sofa::type::fixed_array<long long, 2> >::Inner_Template_Class<unsigned int>;
    EXPECT_EQ(getDecodeTypeName<B>(), "Inner_Template_Class<unsigned int>");
    EXPECT_EQ(getDecodeClassName<B>(), "Inner_Template_Class");
    EXPECT_EQ(getDecodeNamespaceName<B>(), "sofa::__sofa__");
    EXPECT_EQ(getDecodeTemplateName<B>(), "unsigned int");

    using C = sofa::__sofa__::OuterTemplateClass<sofa::type::fixed_array<_UnderscoreClass, 2> >::Inner_Template_Class<sofa::type::Mat<3, 3, double> >;
    EXPECT_EQ(getDecodeClassName<C>(), "Inner_Template_Class");
    EXPECT_EQ(getDecodeNamespaceName<C>(), "sofa::__sofa__");
}