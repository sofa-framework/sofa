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
#include <sofa/gui/common/config.h>
#include <sofa/gui/common/MouseOperations.h>

#include <iostream>
#include <map>

namespace sofa::gui::common
{

class OperationCreator
{
public:
    virtual ~OperationCreator() {}
    virtual Operation* create() const =0;
    virtual std::string getDescription() const=0;
};

template<class RealOperation>
class TOperationCreator: public OperationCreator
{
public:
    Operation* create() const override {return new RealOperation();}
    std::string getDescription() const override { return RealOperation::getDescription();}
};

class SOFA_GUI_COMMON_API OperationFactory
{
public:
    typedef std::map< std::string, OperationCreator* > RegisterStorage;
    RegisterStorage registry;

    static OperationFactory* getInstance()
    {
        static OperationFactory instance;
        return &instance;
    }

    static std::string GetDescription(const std::string &name)
    {
        const RegisterStorage &reg = getInstance()->registry;
        const RegisterStorage::const_iterator it = reg.find(name);
        if (it != reg.end())
        {
            return it->second->getDescription();
        }
        else return std::string();

    }

    SOFA_ATTRIBUTE_DISABLED__TYPO()
    static Operation* Instanciate(const std::string &name) = delete;

    static Operation* Instantiate(const std::string &name)
    {
        const RegisterStorage &reg = getInstance()->registry;
        const RegisterStorage::const_iterator it = reg.find(name);
        if (it != reg.end())
        {
            const OperationCreator *creator=it->second;
            Operation* op=creator->create();
            if (op) op->id=name;
            return op;
        }
        else return nullptr;
    }

};

class SOFA_GUI_COMMON_API RegisterOperation
{
public:
    std::string name;
    OperationCreator *creator { nullptr };

    RegisterOperation(const std::string &n)
    {
        name = n;
    }

    template <class TOperation>
    int add()
    {
        creator = new TOperationCreator< TOperation >();
        OperationFactory::getInstance()->registry.insert(std::make_pair(name, creator));
        return 0; // we return an int so that this method can be called from static variable initializers
    }
};


} // namespace sofa::gui::common
