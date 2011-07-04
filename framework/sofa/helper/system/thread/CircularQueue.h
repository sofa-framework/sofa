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

#ifndef SOFA_HELPER_SYSTEM_THREAD_CIRCULARQUEUE_H
#define SOFA_HELPER_SYSTEM_THREAD_CIRCULARQUEUE_H

#include "../atomic.h"
#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

/**
 * This class implements a fixed-size circular queue.
 * The template parameter ThreadAccessPolicy allows to customize access to the
 * array according to thread-safety requirements.
 */
template<class T, unsigned N, class ThreadAccessPolicy>
class CircularQueue : public ThreadAccessPolicy
{
public:
    CircularQueue();
    ~CircularQueue();

    bool pop(T& item);
    bool push(T& item);

    bool isFull() const;

private:
    helper::fixed_array<T, N> array;
};

/**
 * This is a lock-free single-producer single-consumer implementation of a
 * circular queue matching the ThreadAccessPolicy of CircularQueue.
 */
class OneThreadPerEnd
{
public:
    bool isEmpty() const;

protected:
    OneThreadPerEnd();

    bool isFull(unsigned size) const;

    template<class T>
    void init(T array[], unsigned size);

    template<class T>
    bool pop(T array[], unsigned size, T& item);

    template<class T>
    bool push(T array[], unsigned size, T& item);

    volatile unsigned head;
    volatile unsigned tail;
};

/**
 * This is a lock-free (but not wait-free) multi-producer multi-consumer
 * implementation of a circular queue matching the ThreadAccessPolicy of CircularQueue.
 * It is not recommended to use it when the queue can be full or empty for a
 * long time with threads waiting (because of active waits).
 */
class ManyThreadsPerEnd
{
public:
    bool isEmpty() const;

    typedef helper::system::atomic<int> AtomicInt;
protected:
    ManyThreadsPerEnd();

    bool isFull(int size) const;
    void init(AtomicInt array[], int size);
    bool pop(AtomicInt array[], int size, AtomicInt& item);
    bool push(AtomicInt array[], int size, AtomicInt& item);

    AtomicInt head;
    AtomicInt tail;
};

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_SYSTEM_THREAD_CIRCULARQUEUE_H
