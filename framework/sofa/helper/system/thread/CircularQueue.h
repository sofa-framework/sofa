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
template<class T, template<class T> class StoragePolicy, class ThreadAccessPolicy>
class CircularQueue : public StoragePolicy<T>, public ThreadAccessPolicy
{
public:
    CircularQueue();
    ~CircularQueue();

    bool pop(T& item);
    bool push(const T& item);

    bool isFull() const;
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
class OneThreadPerEnd
{
public:
    bool isEmpty() const;

    unsigned size() const;
protected:
    OneThreadPerEnd();

    bool isFull(unsigned maxSize) const;

    template<class T>
    void init(T array[], unsigned maxCapacity);

    template<class T>
    bool pop(T array[], unsigned maxSize, unsigned maxCapacity, T& item);

    template<class T>
    bool push(T array[], unsigned maxSize, unsigned maxCapacity, const T& item);

    volatile unsigned head;
    volatile unsigned tail;
};

/**
 * This is a lock-free (but not wait-free) multi-producer multi-consumer
 * implementation of a circular queue matching the ThreadAccessPolicy of CircularQueue.
 * @note maxCapacity parameters MUST always be a power of 2.
 */
class ManyThreadsPerEnd
{
public:
    bool isEmpty() const;
    int size() const;

protected:
    typedef helper::system::atomic<int> AtomicInt;

    ManyThreadsPerEnd();

    bool isFull(int size) const;
    void init(AtomicInt array[], int maxCapacity);
    bool pop(AtomicInt array[], int maxSize, int maxCapacity, AtomicInt& item);
    bool push(AtomicInt array[], int maxSize, int maxCapacity, const AtomicInt& item);

    AtomicInt head;
    AtomicInt tail;
};

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_SYSTEM_THREAD_CIRCULARQUEUE_H
