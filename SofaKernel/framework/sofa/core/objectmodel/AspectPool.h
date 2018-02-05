/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#ifndef SOFA_CORE_OBJECTMODEL_ASPECTPOOL_H
#define SOFA_CORE_OBJECTMODEL_ASPECTPOOL_H

#include <sofa/core/ExecParams.h>
#include <sofa/helper/system/atomic.h>
#include <sofa/helper/system/thread/CircularQueue.h>
#include <sofa/helper/vector.h>
#include <sofa/core/sptr.h>

#include <functional>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Aspect;
class AspectPool;
class AspectBuffer;

using AspectRef = sptr<Aspect>;

SOFA_CORE_API void intrusive_ptr_add_ref(Aspect* b);
SOFA_CORE_API void intrusive_ptr_release(Aspect* b);

/**
 * This class represents an allocated aspect.
 * AspectPool returns a smart pointer to an object of this class to give the
 * aspect ownership to the caller.
 * It is safe to use this class from several threads.
 */
class SOFA_CORE_API Aspect
{
public:

    // Only AspectPool is allowed to create and destroy aspects
    friend class AspectPool;

    int aspectID() const { return id; }

    /// Add a reference to this aspect.
    /// Note that you should avoid using this method directly, use AspectRef instead to handle it automatically
    void add_ref();

    /// Release a reference to this aspect.
    /// Note that you should avoid using this method directly, use AspectRef instead to handle it automatically
    void release();

private:
    Aspect(AspectPool& pool, int id);
    ~Aspect();

    AspectPool& pool;
    const int id;
    helper::system::atomic<int> counter;
};



/**
 * This class is responsible for managing the pool of available aspects numbers.
 * It is safe to use this class from several thread.
 */
class SOFA_CORE_API AspectPool
{
public:
    AspectPool();
    ~AspectPool();

    void setReleaseCallback(const std::function<void (int)>& callback);

    /**
     * Request a new aspect.
     * The returned object should stay alive as long as the aspect is in use.
     * It it possible to duplicate the AspectRef if several threads/algorithm use
     * the same aspect.
     * If no aspect remains available, null pointer is returned.
     */
    AspectRef allocate();

    AspectRef getAspect(int id);

    int nbAspects() const { return (int) aspects.size(); }
    int getAspectCounter(int id) const { return aspects[id]->counter; }

    friend class Aspect;

protected:
    /**
     * Release the aspect having the specified number.
     * It makes the number immediately available to satisfy subsequent AspectPool::allocate
     * requests.
     */
    void release(int id);

private:
    AspectPool(const AspectPool& r);
    AspectPool& operator=(const AspectPool& r);

    typedef helper::system::atomic<int> AtomicInt;
    typedef helper::system::thread::CircularQueue<
    AtomicInt,
    helper::system::thread::FixedPower2Size<SOFA_DATA_MAX_ASPECTS>::type,
    helper::system::thread::ManyThreadsPerEnd>
    AspectQueue;

    helper::vector<Aspect*> aspects;
    AspectQueue freeAspects;
    std::function<void (int)> releaseCallback;
};

/**
 * This class is responsible for providing a buffer for communicating aspects between threads,
 * such that only the most up to date aspect is kept, and the previous one is reused to send
 * the next update. This is similar to triple buffering.
 */
class SOFA_CORE_API AspectBuffer
{
public:
    AspectBuffer(AspectPool& pool);
    ~AspectBuffer();

    /// Allocate an aspect ID to prepare the next version, reusing a recent one if possible
    AspectRef allocate();
    /// Send a new version, overriding the latest if it was not already received (in which case it can be "recycled" using allocate)
    void push(AspectRef id);
    /// Receive the latest version, return true if one is available, or false otherwise (in which case id is unchanged)
    bool pop(AspectRef& id);

    /// Clear the buffers
    /// This must be called before either the AspectPool or this buffer is deleted
    void clear();

protected:
    typedef helper::system::atomic<int> AtomicInt;

    AspectPool& pool;
    AtomicInt latestID; ///< -1 or aspect ID of the last version sent
    AtomicInt availableID; ///< -1 or aspect ID available to send the next version

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif /* SOFA_CORE_OBJECTMODEL_ASPECTPOOL_H */
