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

#ifndef SOFA_CORE_OBJECTMODEL_ASPECTPOOL_H
#define SOFA_CORE_OBJECTMODEL_ASPECTPOOL_H

#include <sofa/core/ExecParams.h>
#include <sofa/helper/system/thread/CircularQueue.h>
#include <boost/smart_ptr/intrusive_ptr.hpp>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Aspect;
class AspectPool;
typedef boost::intrusive_ptr<Aspect> AspectRef;

/**
 * This class represents an allocated aspect.
 * AspectPool returns a smart pointer to an object of this class to give the
 * aspect ownership to the caller.
 * It is safe to use this class from several threads.
 */
class Aspect
{
public:
    ~Aspect();

    int aspectID() { return id; }

    friend class AspectPool;
private:
    static AspectRef create(AspectPool* pool, int id);

    Aspect(AspectPool& pool, int id);
    void releaseFromPool();

    AspectPool& pool;
    int id;
    helper::system::atomic<int> counter;

    friend void intrusive_ptr_add_ref(Aspect* b);
    friend void intrusive_ptr_release(Aspect* b);
};



/**
 * This class is responsible for managing the pool of available aspects numbers.
 * It is safe to use this class from several thread.
 */
class AspectPool
{
public:
    AspectPool();
    ~AspectPool();

    AspectRef allocate();

    friend class Aspect;
protected:
    void release(int id);

private:
    AspectPool(const AspectPool& r);
    AspectPool& operator=(const AspectPool& r);

    typedef helper::system::atomic<int> AtomicInt;
    typedef helper::system::thread::CircularQueue<AtomicInt, SOFA_DATA_MAX_INSTANCES, helper::system::thread::ManyThreadsPerEnd> AspectQueue;

    AspectQueue freeAspects;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif /* SOFA_CORE_OBJECTMODEL_ASPECTPOOL_H */
