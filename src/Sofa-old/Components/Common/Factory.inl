#ifndef SOFA_COMPONENTS_COMMON_FACTORY_INL
#define SOFA_COMPONENTS_COMMON_FACTORY_INL

#include "Factory.h"
#include <iostream>
#include <typeinfo>

// added by Sylvere F.
// this inclusion must be done but not in this part of code. For the moment, I don't know where ;)
#include <string>

namespace Sofa
{

namespace Components
{

namespace Common
{

template <typename TKey, class TObject, typename TArgument>
TObject* Factory<TKey, TObject, TArgument>::createObject(Key key, Argument arg)
{
    Object* object;
    Creator* creator;
    typename std::multimap<Key, Creator*>::iterator it = registry.lower_bound(key);
    typename std::multimap<Key, Creator*>::iterator end = registry.upper_bound(key);
    while (it != end)
    {
        creator = (*it).second;
        object = creator->createInstance(arg);
        if (object != NULL)
        {
            //std::cout<<"Object type "<<key<<" created: "<<gettypename(typeid(*object))<<std::endl;
            return object;
        }
        ++it;
    }
    std::cerr<<"Object type "<<key<<" creation failed."<<std::endl;
    return NULL;
}

template <typename TKey, class TObject, typename TArgument>
Factory<TKey, TObject, TArgument>* Factory<TKey, TObject, TArgument>::getInstance()
{
    static Factory<Key, Object, Argument> instance;
    return &instance;
}

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
