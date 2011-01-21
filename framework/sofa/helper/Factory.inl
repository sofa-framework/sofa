/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_FACTORY_INL
#define SOFA_HELPER_FACTORY_INL

#include <sofa/helper/Factory.h>
#include <iostream>
#include <typeinfo>

// added by Sylvere F.
// this inclusion must be done but not in this part of code. For the moment, I don't know where ;)
#include <string>

namespace sofa
{

namespace helper
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
            /*
            std::cout<<"Object type "<<key<<" created: "<<gettypename(typeid(*object))<<std::endl;*/
            return object;
        }
        ++it;
    }
//	std::cerr<<"Object type "<<key<<" creation failed."<<std::endl;
    return NULL;
}

template <typename TKey, class TObject, typename TArgument>
TObject* Factory<TKey, TObject, TArgument>::createAnyObject(Argument arg)
{
    Object* object;
    Creator* creator;
    typename std::multimap<Key, Creator*>::iterator it = registry.begin();
    typename std::multimap<Key, Creator*>::iterator end = registry.end();
    while (it != end)
    {
        creator = (*it).second;
        object = creator->createInstance(arg);
        if (object != NULL)
        {
            return object;
        }
        ++it;
    }
//	std::cerr<<"Object type "<<key<<" creation failed."<<std::endl;
    return NULL;
}


template <typename TKey, class TObject, typename TArgument>
template< typename OutIterator >
void Factory<TKey, TObject, TArgument>::uniqueKeys(OutIterator out)
{

    typename std::multimap<Key, Creator*>::iterator it;

    const Key* p_key = NULL;
    for ( it = registry.begin(); it != registry.end(); ++it)
    {

        if( p_key && *p_key == it->first ) continue;

        p_key = &(it->first);
        *out = *p_key;
        out++;
    }
}

template <typename TKey, class TObject, typename TArgument>
bool Factory<TKey, TObject, TArgument>::hasKey(Key key)
{
    return (this->registry.find(key) != this->registry.end());
}

template <typename TKey, class TObject, typename TArgument>
Factory<TKey, TObject, TArgument>* Factory<TKey, TObject, TArgument>::getInstance()
{
    static Factory<Key, Object, Argument> instance;
    return &instance;
}

} // namespace helper

} // namespace sofa

#endif
