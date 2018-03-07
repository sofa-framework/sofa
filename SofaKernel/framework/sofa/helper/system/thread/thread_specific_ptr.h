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
#ifndef SOFA_HELPER_SYSTEM_THREAD_THREAD_SPECIFIC_PTR_H
#define SOFA_HELPER_SYSTEM_THREAD_THREAD_SPECIFIC_PTR_H

#include <sofa/helper/system/config.h>


#if defined(__GNUC__) && (defined(__linux__) || defined(WIN32))
// use __thread
#define SOFA_TLS_KEYWORD __thread
#elif defined(WIN32)
// use __declspec(thread)
#define SOFA_TLS_KEYWORD __declspec(thread)
#elif defined(_XBOX)
// might be supported, but not needed yet!
#define SOFA_TLS_KEYWORD
#else
// use pthread API
#include <pthread.h>
#define SOFA_TLS_PTHREAD
#endif

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

template<class T> class thread_specific_ptr;

#if defined(SOFA_TLS_KEYWORD)

#define SOFA_THREAD_SPECIFIC_PTR(type,name) static SOFA_TLS_KEYWORD type * name = 0

#elif defined(SOFA_TLS_PTHREAD)

#define SOFA_THREAD_SPECIFIC_PTR(type,name) static ::sofa::helper::system::thread::thread_specific_ptr<type> name

template<class T>
class thread_specific_ptr
{
private:
    pthread_key_t key;

    thread_specific_ptr(thread_specific_ptr&); // NO COPY
    thread_specific_ptr& operator=(thread_specific_ptr&); // NO ASSIGNEMENT

    T* get() const
    {
        void *ptr = pthread_getspecific(key);
        /*
                if (!ptr)
                {
                    ptr = new T;
                    pthread_setspecific(key,ptr);
                }
        */
        return static_cast<T*>(ptr);
    }

    static void destructor(void*)
    {
    }

public:
    thread_specific_ptr()
    {
        pthread_key_create(&key, destructor);
    }
    ~thread_specific_ptr()
    {
        pthread_key_delete(key);
    }
    operator T*() const
    {
        return get();
    }
    T* operator=(T* ptr)
    {
        pthread_setspecific(key,ptr);
        return ptr;
    }
    T* operator->() const
    {
        return get();
    }
    T& operator*() const
    {
        return *get();
    }
};
#else
#error thread local storage is not supported on your platform
#endif

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

#endif
