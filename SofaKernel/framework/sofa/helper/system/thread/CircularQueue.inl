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

#include <sofa/helper/helper.h>
#include "CircularQueue.h"

#include <thread>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::CircularQueue()
{
    ThreadAccessPolicy::init(this->getQueue(), this->maxCapacity());
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::~CircularQueue()
{
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::isEmpty() const
{
    return ThreadAccessPolicy::isEmpty(this->maxSize());
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::isFull() const
{
    return ThreadAccessPolicy::isFull(this->maxSize());
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
unsigned CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::size() const
{
    return ThreadAccessPolicy::size(this->maxSize());
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::push(const T& item)
{
    return ThreadAccessPolicy::push(this->getQueue(), this->maxSize(), this->maxCapacity(), item);
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
bool CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::pop(T& item, bool clear)
{
    return ThreadAccessPolicy::pop(this->getQueue(), this->maxSize(), this->maxCapacity(), item, clear);
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
unsigned CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::skip(unsigned maxsize, bool clear)
{
    return ThreadAccessPolicy::skip(this->getQueue(), this->maxSize(), this->maxCapacity(), maxsize, clear);
}

template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
template<class OutputIterator>
unsigned CircularQueue<T, StoragePolicy, ThreadAccessPolicy>::pop(OutputIterator out, unsigned maxsize, bool clear)
{
    return ThreadAccessPolicy::pop(this->getQueue(), this->maxSize(), this->maxCapacity(), out, maxsize, clear);
}



inline SOFA_HELPER_API OneThreadPerEnd::OneThreadPerEnd()
    : head(0), tail(0)
{
}

inline SOFA_HELPER_API bool OneThreadPerEnd::isEmpty(unsigned /*maxSize*/) const
{
    return head == tail;
}

inline SOFA_HELPER_API bool OneThreadPerEnd::isFull(unsigned maxSize) const
{
    return (tail+1) % maxSize == head;
}

inline SOFA_HELPER_API unsigned OneThreadPerEnd::size(unsigned maxSize) const
{
    unsigned h = head;
    unsigned t = tail;
    if (t < h) t += maxSize;
    return t - h;
}

template<class T>
void OneThreadPerEnd::init(T /*array*/[], unsigned /*maxCapacity*/)
{
}

template<class T>
bool OneThreadPerEnd::push(T array[], unsigned maxSize, unsigned /*maxCapacity*/, const T& item)
{
    unsigned nextTail = (tail + 1) % maxSize;
    if(nextTail == head)
    {
        // queue is full
        return false;
    }
    array[tail] = item;
    tail = nextTail;
    return true;

}

template<class T>
bool OneThreadPerEnd::pop(T array[], unsigned maxSize, unsigned /*maxCapacity*/, T& item, bool clear)
{
    if(isEmpty(maxSize))
    {
        // queue is empty
        return false;
    }

    item = array[head];
    if (clear)
    {
        array[head] = T();
    }
    head = (head + 1) % maxSize;
    return true;
}

template<class T>
unsigned OneThreadPerEnd::skip(T array[], unsigned maxSize, unsigned /*maxCapacity*/, unsigned outmaxsize, bool clear)
{
    unsigned currentSize = size(maxSize);
    unsigned outsize = (currentSize < outmaxsize) ? currentSize : outmaxsize;
    if (outsize == 0)
    {
        // queue is empty
        return 0;
    }
    if (clear)
    {
        unsigned i    = head;
        unsigned iend = head + outsize;
        if (iend >= maxSize)
        {
            for (;i<maxSize;++i)
            {
                array[i] = T();
            }
            i = 0;
            iend -= maxSize;
        }
        for (;i<iend;++i)
        {
            array[i] = T();
        }
        head = iend;
    }
    else
    {
        head = (head + outsize) % maxSize;
    }
    return outsize;
}

template<class T, class OutputIterator>
unsigned OneThreadPerEnd::pop(T array[], unsigned maxSize, unsigned /*maxCapacity*/, OutputIterator out, unsigned outmaxsize, bool clear)
{
    unsigned currentSize = size(maxSize);
    unsigned outsize = (currentSize < outmaxsize) ? currentSize : outmaxsize;
    if (outsize == 0)
    {
        // queue is empty
        return 0;
    }
    // copy values + optional clear
    {
        unsigned i    = head;
        unsigned iend = head + outsize;
        if (iend >= maxSize)
        {
            for (;i<maxSize;++i)
            {
                *out = array[i];
                ++out;
                if (clear)
                {
                    array[i] = T();
                }
            }
            i = 0;
            iend -= maxSize;
        }
        for (;i<iend;++i)
        {
            *out = array[i];
            ++out;
            if (clear)
            {
                array[i] = T();
            }
        }
        head = iend;
    }
    return outsize;
}



inline SOFA_HELPER_API ManyThreadsPerEnd::ManyThreadsPerEnd()
    : head(0), tail(0)
{
}

inline SOFA_HELPER_API bool ManyThreadsPerEnd::isEmpty(unsigned /*maxSize*/) const
{
    return head == tail;
}

inline SOFA_HELPER_API bool ManyThreadsPerEnd::isFull(unsigned maxSize) const
{
    return tail-head >= (int)maxSize;
}

inline SOFA_HELPER_API unsigned ManyThreadsPerEnd::size(unsigned /*maxSize*/) const
{
    return tail - head;
}

inline SOFA_HELPER_API void ManyThreadsPerEnd::init(AtomicInt array[], int maxCapacity)
{
    for(int i = 0; i < maxCapacity; ++i)
    {
        array[i] = -1;
    }
}

inline SOFA_HELPER_API bool ManyThreadsPerEnd::pop(AtomicInt array[], int maxSize, int maxCapacity, AtomicInt& item, bool /*clear*/)
{
    if(isEmpty(maxSize))
    {
        std::this_thread::yield();
        return false;
    }
    // atomically reserve the element to read
    int readIdx = head.exchange_and_add(1) & (maxCapacity-1); // maxCapacity is assumed to be a power of 2

    // Active wait:
    // loop as long as other threads have not put any valid value in the element.
    // It happens when the queue is temporarily empty.
    while((item = array[readIdx]) == -1)
    {
    }

    // mark the element as available
    array[readIdx] = -1;

    return true;
}

inline SOFA_HELPER_API bool ManyThreadsPerEnd::push(AtomicInt array[], int maxSize, int maxCapacity, const AtomicInt& item)
{
    if(isFull(maxSize))
    {
        std::this_thread::yield();
        return false;
    }

    // atomically reserve the element to write
    int writeIdx = tail.exchange_and_add(1) & (maxCapacity-1); // maxCapacity is assumed to be a power of 2
    // Active wait:
    // loop as long as the element has not been read by another thread (which is indicated by a -1 value).
    // It happens when the queue is temporarily full.
    while(array[writeIdx].compare_and_swap(-1, item) != -1)
    {
    }

    return true;
}

inline SOFA_HELPER_API unsigned ManyThreadsPerEnd::skip(AtomicInt array[], int maxSize, int maxCapacity, unsigned outmaxsize, bool clear)
{
    // NOT OPTIMIZED
    unsigned outsize = 0;
    AtomicInt tmp;
    while (outsize < outmaxsize && pop(array, maxSize, maxCapacity, tmp, clear))
    {
        ++outsize;
    }
    return outsize;
}

template<class OutputIterator>
unsigned ManyThreadsPerEnd::pop(AtomicInt array[], int maxSize, int maxCapacity, OutputIterator out, unsigned outmaxsize, bool clear)
{
    // NOT OPTIMIZED
    unsigned outsize = 0;
    while (outsize < outmaxsize && pop(array, maxSize, maxCapacity, *out, clear))
    {
        ++out;
        ++outsize;
    }
    return outsize;
}

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa
