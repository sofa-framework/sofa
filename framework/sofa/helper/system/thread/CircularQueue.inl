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

#include "CircularQueue.h"

#if SOFA_HAVE_BOOST
#include <boost/thread/thread.hpp>
#endif

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

template<class T, unsigned N, class ThreadAccessPolicy>
CircularQueue<T, N, ThreadAccessPolicy>::CircularQueue()
{
    ThreadAccessPolicy::init(&array[0], N);
}

template<class T, unsigned N, class ThreadAccessPolicy>
CircularQueue<T, N, ThreadAccessPolicy>::~CircularQueue()
{
}

template<class T, unsigned N, class ThreadAccessPolicy>
bool CircularQueue<T, N, ThreadAccessPolicy>::isFull() const
{
    return ThreadAccessPolicy::isFull(N);
}

template<class T, unsigned N, class ThreadAccessPolicy>
bool CircularQueue<T, N, ThreadAccessPolicy>::pop(T& item)
{
    return ThreadAccessPolicy::pop(&array[0], N, item);
}

template<class T, unsigned N, class ThreadAccessPolicy>
bool CircularQueue<T, N, ThreadAccessPolicy>::push(const T& item)
{
    return ThreadAccessPolicy::push(&array[0], N, item);
}



OneThreadPerEnd::OneThreadPerEnd()
    : head(0), tail(0)
{
}

bool OneThreadPerEnd::isEmpty() const
{
    return head == tail;
}

bool OneThreadPerEnd::isFull(unsigned size) const
{
    return (tail+1) % size == head;
}

unsigned OneThreadPerEnd::size() const
{
    return tail - head;
}

template<class T>
void OneThreadPerEnd::init(T /*array*/[], unsigned /*size*/)
{
}

template<class T>
bool OneThreadPerEnd::pop(T array[], unsigned size, T& item)
{
    if(isEmpty()) return false;

    item = array[head];
    head = (head + 1) % size;
    return true;
}

template<class T>
bool OneThreadPerEnd::push(T array[], unsigned size, const T& item)
{
    unsigned nextTail = (tail + 1) % size;
    if(nextTail != head)
    {
        array[tail] = item;
        tail = nextTail;
        return true;
    }

    // queue was full
    return false;
}



ManyThreadsPerEnd::ManyThreadsPerEnd()
    : head(0), tail(0)
{
}

bool ManyThreadsPerEnd::isEmpty() const
{
    return head >= tail;
}

bool ManyThreadsPerEnd::isFull(int size) const
{
    return (tail - head + 1) >= size;
}


int ManyThreadsPerEnd::size() const
{
    return tail - head;
}

void ManyThreadsPerEnd::init(AtomicInt array[], int size)
{
    for(int i = 0; i < size; ++i)
    {
        array[i] = -1;
    }
}

bool ManyThreadsPerEnd::pop(AtomicInt array[], int size, AtomicInt& item)
{
    if(isEmpty())
    {
#if SOFA_HAVE_BOOST
        boost::thread::yield();
#endif
        return false;
    }
    // atomically reserve the element to read
    int readIdx = head.exchange_and_add(1) % size;

    // Active wait:
    // loop as long as other threads have not put any valid value in the element.
    // It happens when the queue is temporarily empty.
    while((item = array[readIdx]) == -1);

    // mark the element as available
    array[readIdx] = -1;

    return true;
}

bool ManyThreadsPerEnd::push(AtomicInt array[], int size, const AtomicInt& item)
{
    if(isFull(size))
    {
#if SOFA_HAVE_BOOST
        boost::thread::yield();
#endif
        return false;
    }

    // atomically reserve the element to write
    int writeIdx = tail.exchange_and_add(1) % size;

    // Active wait:
    // loop as long as the element has not been read by another thread (which is indicated by a -1 value).
    // It happens when the queue is temporarily full.
    while(array[writeIdx].compare_and_swap(-1, item) != -1);

    return true;
}

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa
