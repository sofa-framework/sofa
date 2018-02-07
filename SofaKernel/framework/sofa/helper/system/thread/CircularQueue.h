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
 * This class implements a policy-based circular queue.
 * The template parameter ThreadAccessPolicy allows to customize access to the
 * array according to thread-safety requirements.
 */
template<class T, template<class U> class StoragePolicy, class ThreadAccessPolicy>
class CircularQueue : public StoragePolicy<T>, public ThreadAccessPolicy
{
public:
    CircularQueue();
    ~CircularQueue();

    bool push(const T& item);
    bool pop(T& item, bool clear = true);

    bool isEmpty() const;
    unsigned size() const;
    bool isFull() const;

    unsigned skip(unsigned maxsize, bool clear = true);
    template<class OutputIterator>
    unsigned pop(OutputIterator out, unsigned maxsize, bool clear = true);
};

template<int N>
struct AlignPow2
{
    enum { prev = AlignPow2<N/2>::val*2 };
    enum { val = (prev >= N ? prev : prev*2) };
};

template<>
struct AlignPow2<1>
{
    enum { val = 1 };
};

template<>
struct AlignPow2<0>
{
    enum { val = 1 };
};

/**
 * This is a storage policy for CircularQueue that uses a compile-time fixed-size array.
 */
template<int N>
struct FixedSize
{
    template<class T>
    class type
    {
    public:
        T* getQueue() { return &array[0]; }

        static int maxSize() { return N; }
        static int maxCapacity() { return N; }

    private:
        helper::fixed_array<T, N> array;
    };
};

/**
 * This is a storage policy for CircularQueue that uses a compile-time fixed-size
 * array aligned on the upper power of 2 of the specified template parameter.
 */
template<int N>
struct FixedPower2Size
{
    template<class T>
    class type
    {
    public:
        T* getQueue() { return &array[0]; }

        static int maxSize() { return MaxSize; }
        static int maxCapacity() { return MaxCapacity; }

    private:
        enum { MaxSize = N };
        enum { MaxCapacity = AlignPow2<MaxSize>::val };

        helper::fixed_array<T, MaxCapacity> array;
    };
};

/**
 * This is a lock-free single-producer single-consumer implementation of a
 * circular queue matching the ThreadAccessPolicy of CircularQueue.
 */
class SOFA_HELPER_API OneThreadPerEnd
{
public:

protected:
    OneThreadPerEnd();

    bool isEmpty(unsigned maxSize) const;

    bool isFull(unsigned maxSize) const;

    unsigned size(unsigned maxSize) const;

    template<class T>
    void init(T array[], unsigned maxCapacity);

    template<class T>
    bool push(T array[], unsigned maxSize, unsigned maxCapacity, const T& item);
    
    template<class T>
    bool pop(T array[], unsigned maxSize, unsigned maxCapacity, T& item, bool clear = true);

    template<class T>
    unsigned skip(T array[], unsigned maxSize, unsigned maxCapacity, unsigned outmaxsize, bool clear = true);
    
    template<class T, class OutputIterator>
    unsigned pop(T array[], unsigned maxSize, unsigned maxCapacity, OutputIterator out, unsigned outmaxsize, bool clear = true);
    
    volatile unsigned head;
    volatile unsigned tail;
};

/**
 * This is a lock-free (but not wait-free) multi-producer multi-consumer
 * implementation of a circular queue matching the ThreadAccessPolicy of CircularQueue.
 * @note maxCapacity parameters MUST always be a power of 2.
 */
class SOFA_HELPER_API ManyThreadsPerEnd
{
public:

protected:
    typedef helper::system::atomic<int> AtomicInt;

    ManyThreadsPerEnd();
    
    bool isEmpty(unsigned maxSize) const;

    bool isFull(unsigned maxSize) const;

    unsigned size(unsigned maxSize) const;

    void init(AtomicInt array[], int maxCapacity);

    bool push(AtomicInt array[], int maxSize, int maxCapacity, const AtomicInt& item);

    bool pop(AtomicInt array[], int maxSize, int maxCapacity, AtomicInt& item, bool clear = true);

    unsigned skip(AtomicInt array[], int maxSize, int maxCapacity, unsigned outmaxsize, bool clear = true);
    
    template<class OutputIterator>
    unsigned pop(AtomicInt array[], int maxSize, int maxCapacity, OutputIterator out, unsigned outmaxsize, bool clear = true);

    AtomicInt head;
    AtomicInt tail;
};

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_SYSTEM_THREAD_CIRCULARQUEUE_H
