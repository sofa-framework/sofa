/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "AspectPool.h"
#include <sofa/helper/system/thread/CircularQueue.inl>
#include <iostream>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 * Creates a new AspectRef.
 * This method is private to only allow AspectPool to use it.
 */
AspectRef Aspect::create(AspectPool* pool, int id)
{
    AspectRef ref(new Aspect(*pool, id));
    return ref;
}

/**
 * Constructor.
 * It is private to only allow Aspect::create to use it.
 */
Aspect::Aspect(AspectPool& pool, int id)
    : pool(pool), id(id), counter(0)
{
}

/**
 * Destructor: release the aspect from the pool.
 */
Aspect::~Aspect()
{
    releaseFromPool();
}

/**
 * Release this aspect from the pool.
 */
void Aspect::releaseFromPool()
{
    pool.release(id);
}

void intrusive_ptr_add_ref(Aspect* a)
{
    a->counter.inc();
}
void intrusive_ptr_release(Aspect* a)
{
    if(a->counter.dec_and_test_null())
    {
        delete a;
    }
}


/**
 * Constructor: creates a new aspect pool.
 */
AspectPool::AspectPool()
{
    // Fill the list of free aspects.
    std::cout << "AspectPool"<<this<<": filling " << SOFA_DATA_MAX_ASPECTS << " aspects" << std::endl;
    for(int i = 0; i < SOFA_DATA_MAX_ASPECTS; ++i)
    {
        AtomicInt aspectID(i);
        freeAspects.push(aspectID);
    }
    std::cout << "AspectPool"<<this<<": " << freeAspects.size() << " aspects available" << std::endl;
}

/**
 * Destructor.
 */
AspectPool::~AspectPool()
{
}

void AspectPool::setReleaseCallback(const boost::function<void (int)>& callback)
{
    releaseCallback = callback;
}

/**
 * Request a new aspect.
 * The returned object should stay alive as long as the aspect is in use.
 * It it possible to duplicate the AspectRef if several threads/algorithm use
 * the same aspect.
 * If no aspect remains available, null pointer is returned.
 */
AspectRef AspectPool::allocate()
{
    AspectRef ref;
    AtomicInt aspectID;
    std::cout << "AspectPool"<<this<<": allocate" << std::endl;
    std::cout << "AspectPool"<<this<<": " << freeAspects.size() << " aspects available" << std::endl;
    if(freeAspects.pop(aspectID))
    {
        std::cout << "AspectPool"<<this<<": aspect " << aspectID << " allocated" << std::endl;
        ref = Aspect::create(this, aspectID);
    }
    else
        std::cout << "AspectPool"<<this<<": no aspect available" << std::endl;
    return ref;
}

/**
 * Release the aspect having the specified number.
 * It makes the number immediately available to satisfy subsequent AspectPool::allocate
 * requests.
 */
void AspectPool::release(int id)
{
    std::cout << "AspectPool"<<this<<": release aspect " << id << std::endl;
    if(releaseCallback != 0)
    {
        releaseCallback(id);
    }
    AtomicInt aspectID(id);
    freeAspects.push(aspectID);
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
