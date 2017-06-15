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
#ifndef SOFA_CORE_OBJECTMODEL_CLASSINFO_H
#define SOFA_CORE_OBJECTMODEL_CLASSINFO_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/objectmodel/Data.h>
#include <string>
#include <map>


namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;

/**
 *  \brief Meta information class
 *
 *  This class contains reflection-like features to analyse a class deriving from Base
 *
 */
class SOFA_CORE_API ClassInfo
{
protected:
    const std::type_info* pt;
    ClassInfo(const std::type_info* ti)
        : pt(ti)
    {
        classes[sofa::helper::TypeInfo(*ti)] = this;
    }

    virtual ~ClassInfo()
    {
    }
public:

    std::string name() const { return pt->name(); }
    operator const std::type_info&() const { return *pt; }
    helper::TypeInfo type() const { return sofa::helper::TypeInfo(*pt); }
    bool operator==(const ClassInfo& t) const { return *pt == *t.pt; }
    bool operator!=(const ClassInfo& t) const { return *pt != *t.pt; }
#ifdef _MSC_VER
    bool operator<(const ClassInfo& t) const { return (pt->before(*t.pt)!=0); }
#else
    bool operator<(const ClassInfo& t) const { return pt->before(*t.pt); }
#endif

    virtual void* dynamicCast(Base* obj) const = 0;

    virtual bool isInstance(Base* obj) const
    {
        return dynamicCast(obj) != NULL;
    }
protected:
    static std::map<sofa::helper::TypeInfo, ClassInfo*> classes;
};

template<class T>
class TClassInfo : public ClassInfo
{
protected:
    TClassInfo()
        : ClassInfo(&typeid(T))
    {
    }

public:
    static TClassInfo<T>& get()
    {
        static TClassInfo<T> inst;
        return inst;
    }

    virtual void* dynamicCast(Base* obj) const
    {
        return dynamic_cast<T*>(obj);
    }
};

template<class T>
const ClassInfo& classidT() { return TClassInfo<T>::get(); }

#define classid(T) sofa::core::objectmodel::classidT<T>()

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
