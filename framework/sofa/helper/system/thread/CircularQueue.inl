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

#include "CircularQueue.h"

#if SOFA_HAVE_BOOST
#include <boost/thread/thread.hpp>
#endif
#include <iostream>

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

template<class T, template<class T> class StoragePolicy, class ThreadAccessPolicy>
CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::CircularQueue()
{
    ThreadAccessPolicy::init(this->getQueue(), this->maxCapacity());
}

template<class T, template<class T> class StoragePolicy, class ThreadAccessPolicy>
CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::~CircularQueue()
{
}

template<class T, template<class T> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::isFull() const
{
    return ThreadAccessPolicy::isFull(this->maxSize());
}

template<class T, template<class T> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::pop(T& item)
{
    return ThreadAccessPolicy::pop(this->getQueue(), this->maxSize(), this->maxCapacity(), item);
}

template<class T, template<class T> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::push(const T& item)
{
    return ThreadAccessPolicy::push(this->getQueue(), this->maxSize(), this->maxCapacity(), item);
}



OneThreadPerEnd::OneThreadPerEnd()
    : head(0), tail(0)
{
}

bool OneThreadPerEnd::isEmpty() const
{
    return head == tail;
}

bool OneThreadPerEnd::isFull(unsigned maxSize) const
{
    return (tail+1) % maxSize == head;
}

unsigned OneThreadPerEnd::size() const
{
    return tail - head;
}

template<class T>
void OneThreadPerEnd::init(T /*array*/[], unsigned /*maxCapacity*/)
{
}

template<class T>
bool OneThreadPerEnd::pop(T array[], unsigned maxSize, unsigned /*maxCapacity*/, T& item)
{
    if(isEmpty()) return false;

    item = array[head];
    array[head] = T();
    head = (head + 1) % maxSize;
    return true;
}

template<class T>
bool OneThreadPerEnd::push(T array[], unsigned maxSize, unsigned /*maxCapacity*/, const T& item)
{
    unsigned nextTail = (tail + 1) % maxSize;
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

bool ManyThreadsPerEnd::isFull(int maxSize) const
{
    return tail-head >= maxSize;
}


int ManyThreadsPerEnd::size() const
{
    return tail - head;
}

void ManyThreadsPerEnd::init(AtomicInt array[], int maxCapacity)
{
    for(int i = 0; i < maxCapacity; ++i)
    {
        array[i] = -1;
    }
}

bool ManyThreadsPerEnd::pop(AtomicInt array[], int maxSize, int maxCapacity, AtomicInt& item)
{
    if(isEmpty())
    {
#if SOFA_HAVE_BOOST
        boost::thread::yield();
#endif
        return false;
    }
    // atomically reserve the element to read
    int readIdx = head.exchange_and_add(1) & (maxCapacity-1); // maxCapacity is assumed to be a power of 2
    std::cout << "CQ"<<array<<"/"<<maxSize<<"/"<<maxCapacity<<": pop from " << readIdx << std::endl;

    // Active wait:
    // loop as long as other threads have not put any valid value in the element.
    // It happens when the queue is temporarily empty.
    while((item = array[readIdx]) == -1)
        std::cout << "CQ"<<array<<"/"<<maxSize<<"/"<<maxCapacity<<": pop wait loop" << std::endl;

    // mark the element as available
    array[readIdx] = -1;

    return true;
}

bool ManyThreadsPerEnd::push(AtomicInt array[], int maxSize, int maxCapacity, const AtomicInt& item)
{
    if(isFull(maxSize))
    {
#if SOFA_HAVE_BOOST
        boost::thread::yield();
#endif
        return false;
    }

    // atomically reserve the element to write
    int writeIdx = tail.exchange_and_add(1) & (maxCapacity-1); // maxCapacity is assumed to be a power of 2
    std::cout << "CQ"<<array<<"/"<<maxSize<<"/"<<maxCapacity<<": push " << (int)item << " to " << writeIdx << std::endl;
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
