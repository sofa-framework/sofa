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
#pragma once
#include <sofa/core/objectmodel/BaseLink.h>

namespace sofa::core::objectmodel
{

template<class Owner, class T>
class LinkableContainer : public BaseLink
{
    T& container;
public:
    LinkableContainer(T& c, const std::string& name, const std::string& help, Owner* owner) : BaseLink{ BaseLink::FLAG_STRONGLINK | BaseLink::FLAG_STOREPATH }, container{c}
    {
        setName(name);
        setHelp(help);
        m_owner = owner;
    }

    const BaseClass* getDestClass() const { return T::pointed_type::GetClass(); }
    const BaseClass* getOwnerClass() const { return Owner::GetClass(); }

    Base* getOwnerBase() const override {
        return m_owner; }
    BaseData* getOwnerData() const override { return nullptr;}
    BaseData* getLinkedData(std::size_t=0) const override {return nullptr;}

    virtual void clear(){ container.clear(); }
    virtual bool contains(Base* item){ return container[0] == item ; }
    virtual size_t size() const { return container.size(); }
    virtual size_t getSize() const override { return container.size(); }
    Base* _doGetOwner_() const override {
        return m_owner; }
    void _doSetOwner_(Base* owner) override { m_owner = dynamic_cast<Owner*>(owner); }
    void _doClear_() override { container.clear(); }
    virtual Base* _doGet_(const size_t index) const {
        return container[index]; }
    virtual bool _doAdd_(Base* target, const std::string&) { container.set(dynamic_cast<typename T::pointed_type*>(target)); return true; }
    virtual bool _doRemoveAt_(size_t) { return true; }
    virtual bool _doSet_(Base* target, const size_t index) {
        set(dynamic_cast<typename T::pointed_type*>(const_cast<Base*>(target))); return true; }
    virtual bool _doSet_(Base* target, const std::string&, size_t index=0) {
        container[index] =dynamic_cast<typename T::pointed_type*>(target); return true; }
    virtual bool _isCompatibleOwnerType_(const Base* b) const { return dynamic_cast<typename T::pointed_type*>(const_cast<Base*>(b))==nullptr; }

    std::string _doGetLinkedPath_(const size_t index=0) const override {
        return _doGetPath_(index); }
    virtual std::string _doGetPath_(const size_t index) const {
        if(container[index]==nullptr)
            return "";
        return "@"+container[index]->getPathName();
    }

    Base* m_owner {nullptr};
};

}
