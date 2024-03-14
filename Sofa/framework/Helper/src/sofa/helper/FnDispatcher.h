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
#ifndef SOFA_HELPER_FNDISPATCHER_H
#define SOFA_HELPER_FNDISPATCHER_H

#include <map>

#include <sofa/helper/config.h>
#include <sofa/helper/TypeInfo.h>
namespace sofa
{

namespace helper
{

template <class BaseClass, typename ResulT = void>
class BasicDispatcher
{
public:
    typedef ResulT (*F)(BaseClass &,BaseClass &);

protected:
    typedef std::pair<TypeInfo,TypeInfo> KeyType;
    typedef std::map<KeyType, F> MapType;
    MapType callBackMap;
    virtual ~BasicDispatcher() {}
public:
    void add(const std::type_info& class1, const std::type_info& class2, F fun) ;
    void ignore(const std::type_info& class1, const std::type_info& class2) ;

    template <class ConcreteClass1,class ConcreteClass2,ResulT (*F)(ConcreteClass1&,ConcreteClass2&), bool symetric>
    void ignore()
    {
        this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass1), typeid(ConcreteClass2), &ignoreFn);
        if (symetric)
        {
            this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass2), typeid(ConcreteClass1), &ignoreFn);
        }
    }

    virtual ResulT defaultFn(BaseClass& arg1, BaseClass& arg2);
    static ResulT ignoreFn(BaseClass& arg1, BaseClass& arg2);
    ResulT go(BaseClass &arg1,BaseClass &arg2);
    /// Return true if a pair of argument correspond to a callback function (different than ignoreFn)
    bool isSupported(BaseClass &arg1, BaseClass &arg2);
};

template <class BaseClass, typename ResulT>
class FnDispatcher : public BasicDispatcher<BaseClass, ResulT>
{
public:

    template <class ConcreteClass1, class ConcreteClass2, ResulT (*F)(ConcreteClass1&,ConcreteClass2&), bool symetric>
    void add()
    {
        struct Local
        {
            static ResulT trampoline(BaseClass &arg1, BaseClass &arg2)
            {
                return F(static_cast<ConcreteClass1 &> (arg1),
                    static_cast<ConcreteClass2 &> (arg2));
            }
            static ResulT trampolineR(BaseClass &arg1, BaseClass &arg2)
            {
                return trampoline(arg2, arg1);
            }
        };
        this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass1), typeid(ConcreteClass2), &Local::trampoline);
        if (symetric)
        {
            this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass2), typeid(ConcreteClass1), &Local::trampolineR);
        }
    }

    template <class ConcreteClass1, class ConcreteClass2, bool symetric>
    void ignore()
    {
        this->BasicDispatcher<BaseClass, ResulT>::ignore(typeid(ConcreteClass1), typeid(ConcreteClass2));
        if (symetric)
        {
            this->BasicDispatcher<BaseClass, ResulT>::ignore(typeid(ConcreteClass2), typeid(ConcreteClass1));
        }
    }
};


template <class BaseClass, typename ResulT>
class SingletonFnDispatcher : public FnDispatcher<BaseClass, ResulT>
{
protected:
    SingletonFnDispatcher();
public:
    static SingletonFnDispatcher<BaseClass, ResulT>* getInstance();
};

} // namespace helper

} // namespace sofa

#endif
