/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_FACTORY_H
#define SOFA_HELPER_FACTORY_H

#include <map>
#include <iostream>
#include <typeinfo>
#include <type_traits>

#include <sofa/helper/helper.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

/// Allow us to use BaseCreator and Factory without using any Arguments
class NoArgument {} ;

/// Decode the type's name to a more readable form if possible
std::string SOFA_HELPER_API gettypename(const std::type_info& t);

/// Log classes registered in the factory
void SOFA_HELPER_API logFactoryRegister(std::string baseclass, std::string classname, std::string key, bool multi);

/// Print factory log
void SOFA_HELPER_API printFactoryLog(std::ostream& out = std::cout);

template <class Object, class Argument = NoArgument, class ObjectPtr = Object*>
class BaseCreator
{
public:
    virtual ~BaseCreator() { }
    virtual ObjectPtr createInstance(Argument arg) = 0;
    virtual const std::type_info& type() = 0;
};

template <typename TKey, class TObject, typename TArgument = NoArgument, typename TPtr = TObject* >
class Factory
{
public:
    typedef TKey      Key;
    typedef TObject   Object;
    typedef TPtr      ObjectPtr;
    typedef TArgument Argument;
    typedef BaseCreator<Object, Argument, ObjectPtr> Creator;
    typedef std::multimap<Key, Creator> Registry;

protected:
    std::multimap<Key, Creator*> registry;

public:
    bool registerCreator(Key key, Creator* creator, bool multi=false)
    {
        if(!multi && this->registry.find(key) != this->registry.end())
            return false; // key used
        logFactoryRegister(gettypename(typeid(Object)), gettypename(creator->type()), key, multi);
        this->registry.insert(std::pair<Key, Creator*>(key, creator));
        return true;
    }

    template< class U = Argument, typename std::enable_if<std::is_same<U, NoArgument>::value, int>::type = 0>
    ObjectPtr createObject(Key key, Argument arg = NoArgument()){
        createObject(key, arg);
    }

    ObjectPtr createObject(Key key, Argument arg);

    ObjectPtr createAnyObject(Argument arg);

    template< typename OutIterator >
    void uniqueKeys(OutIterator out);

    bool hasKey(Key key);
    bool duplicateEntry( Key existing, Key duplicate);
    bool resetEntry( Key existingKey);

    static Factory<Key, Object, Argument, ObjectPtr>* getInstance();

    static ObjectPtr CreateObject(Key key, Argument arg)
    {
        return getInstance()->createObject(key, arg);
    }

    static ObjectPtr CreateAnyObject(Argument arg)
    {
        return getInstance()->createAnyObject(arg);
    }

    static bool HasKey(Key key)
    {
        return getInstance()->hasKey(key);
    }

    static bool DuplicateEntry(Key existing,Key duplicate )
    {
        return getInstance()->duplicateEntry(existing, duplicate);
    }

    static bool ResetEntry(Key existing)
    {
        return getInstance()->resetEntry(existing);
    }


    typedef typename std::multimap<Key, Creator*>::iterator iterator;
    iterator begin() { return registry.begin(); }
    iterator end() { return registry.end(); }
    typedef typename std::multimap<Key, Creator*>::const_iterator const_iterator;
    const_iterator begin() const { return registry.begin(); }
    const_iterator end() const { return registry.end(); }
};

template <class Factory, class RealObject>
class Creator : public Factory::Creator, public Factory::Key
{
public:
    typedef typename Factory::Object    Object;
    typedef typename Factory::ObjectPtr ObjectPtr;
    typedef typename Factory::Argument  Argument;
    typedef typename Factory::Key       Key;
    explicit Creator(Key key, bool multi=false)
        : Key(key)
    {
        Factory::getInstance()->registerCreator(key, this, multi);
    }
    ObjectPtr createInstance(Argument arg)
    {
        RealObject* instance = NULL;
        return RealObject::create(instance, arg);
    }
    const std::type_info& type()
    {
        return typeid(RealObject);
    }

    // Dummy function to avoid dead stripping symbol
    void registerInFactory()
    {
        msg_info("Creator") << "[SOFA]Registration of class : " << type().name();
    }

};
/*
/// Generic object creator. Can be specialized for custom objects creation
template<class Object, class Argument>
Object create(Object* obj, Argument arg)
{
    return new Object(arg);
}
*/
template <class Factory, class RealObject>
class CreatorFn : public Factory::Creator, public Factory::Key
{
public:
    typedef typename Factory::Object    Object;
    typedef typename Factory::ObjectPtr ObjectPtr;
    typedef typename Factory::Argument  Argument;
    typedef typename Factory::Key       Key;
    typedef ObjectPtr Fn(RealObject* obj, Argument arg);
    Fn* constructor;

    CreatorFn(Key key, Fn* constructor, bool multi=false)
        : Key(key), constructor(constructor)
    {
        Factory::getInstance()->registerCreator(key, this, multi);
    }

    ObjectPtr createInstance(Argument arg)
    {
        RealObject* instance = NULL;
        return (*constructor)(instance, arg);
    }
    const std::type_info& type()
    {
        return typeid(RealObject);
    }
};


} // namespace helper

} // namespace sofa

// Creator is often used without namespace qualifiers
using sofa::helper::Creator;

#endif
