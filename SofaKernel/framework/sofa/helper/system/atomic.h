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
#ifndef SOFA_HELPER_SYSTEM_ATOMIC_H
#define SOFA_HELPER_SYSTEM_ATOMIC_H

#include <sofa/helper/system/config.h>

#if defined(SOFA_USE_ASM_ATOMIC)
#include <asm/atomic.h>
#include <bits/atomicity.h> // for __exchange_and_add
#elif defined(__GNUC__) && (defined(i386) || defined(__i386__) || defined(__x86_64__))
// custom ASM code, no include
#elif defined(__GNUC__) && !(defined(PS3))
// Fall-back mode: stdc++ atomic operations (should be available on all gcc-supported platforms)
#include <bits/atomicity.h>
#elif defined(WIN32)
#include <windows.h>
#elif defined(_XBOX)
#include <xtl.h>
#elif defined(PS3)
#include <cell/atomic.h>
#else
#error atomic operations are not supported on your platform
#endif

namespace sofa
{

namespace helper
{

namespace system
{

/// Atomic type. Only int is supported for now
template<class T> class atomic;

#if defined(SOFA_USE_ASM_ATOMIC)

/// Small class used for multi-process reference counting
template<> class atomic<int>
{
    atomic_t val;
public:
    atomic() {}
    atomic(int i) { set(i); }
    operator int() const { return atomic_read(&val); }
    void set(int i) { atomic_set(&val,i); }
    void add(int i) { atomic_add(i,&val); }
    void sub(int i) { atomic_sub(i,&val); }
    void inc() { atomic_inc(&val); }
    void dec() { atomic_dec(&val); }
    //bool sub_and_test_null(int i) { return atomic_sub_and_test(i,&val); }
    bool dec_and_test_null() { return atomic_dec_and_test(&val); }
    //bool inc_and_test_null() { return atomic_inc_and_test(&val); }
    bool add_and_test_neg(int i) { return atomic_add_negative(i,&val); }
    atomic& operator= (int i) { set(i); return *this; }
    /// Add a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator+=(int i) { add(i); }
    /// Substract a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator-=(int i) { sub(i); }
    void operator++() { inc(); }
    void operator++(int) { inc(); }
    void operator--() { dec(); }
    void operator--(int) { dec(); }

    int exchange_and_add(int i) { return __exchange_and_add((_Atomic_word*)&val,i); }
    int exchange(int i) { return atomic_xchg(&val,i); }
    int compare_and_swap(int cmp, int with) { return atomic_cmpxchg(&val, cmp, with); }

    static const char* getImplName() { return "Kernel"; }
};

#elif defined(__GNUC__) && (defined(i386) || defined(__i386__) || defined(__x86_64__))

/// Small class used for multi-process reference counting
template<> class atomic<int>
{
    volatile int val;
public:
    atomic() {}
    atomic(int i) { set(i); }
    operator int() const { return val; }
    void set(int i) { val = i; }
    void add(int i)
    {
        __asm__ __volatile__ ("lock add{l} {%1,%0|%0,%1}"
                : "+m" (val) : "ir" (i) : "memory");
    }
    void sub(int i)
    {
        __asm__ __volatile__ ("lock sub{l} {%1,%0|%0,%1}"
                : "+m" (val) : "ir" (i) : "memory");
    }
    void inc()
    {
        __asm__ __volatile__ ("lock inc{l} %0"
                : "+m" (val) : : "memory");
    }
    void dec()
    {
        __asm__ __volatile__ ("lock dec{l} %0"
                : "+m" (val) : : "memory");
    }
    bool dec_and_test_null()
    {
        unsigned char c;
        __asm__ __volatile__("lock dec{l} %0; sete %1"
                :"+m" (val), "=qm" (c)
                : : "memory");
        return c != 0;
    }
    bool add_and_test_neg(int i)
    {
        unsigned char c;
        __asm__ __volatile__("lock add{l} {%2,%0|%0,%2}; sets %1"
                :"+m" (val), "=qm" (c)
                :"ir" (i) : "memory");
        return c != 0;
    }

    int exchange_and_add(int i)
    {
        int res;
        __asm__ __volatile__ ("lock xadd{l} {%0,%1|%1,%0}"
                : "=r" (res), "+m" (val)
                : "0" (i)
                : "memory");
        return res;
    }

    int exchange(int with)
    {
        int result;
        __asm__ __volatile__ ( "lock xchg %0,%1"
                : "=a" (result), "+m" (val)
                : "a" (with)
                : "memory");
        return result;
    }

    int compare_and_swap(int cmp, int with)
    {
        int result;
        __asm__ __volatile__ ( "lock cmpxchg %3,%1"
                : "=a" (result), "+m" (val)
                : "a" (cmp), "r" (with)
                : "memory", "cc");
        return result;
    }

    atomic& operator= (int i) { set(i); return *this; }
    /// Add a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator+=(int i) { add(i); }
    /// Substract a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator-=(int i) { sub(i); }
    void operator++() { inc(); }
    void operator++(int) { inc(); }
    void operator--() { dec(); }
    void operator--(int) { dec(); }

#if defined(__x86_64__)
    static const char* getImplName() { return "x86_64"; }
#else
    static const char* getImplName() { return "386"; }
#endif
};

#elif defined(__GNUC__) && !defined(PS3)
// Fall-back mode: stdc++ atomic operations (should be available on all gcc-supported platforms)

using namespace __gnu_cxx;

/// Small class used for multi-process reference counting
template<> class atomic<int>
{
    volatile _Atomic_word val;
public:
    atomic() {}
    atomic(int i) { set(i); }
    operator int() const
    {
        // this is the correct implementation
        //     return __exchange_and_add(&val,0);
        // but this is faster and should also work on x86 and ppc
        return val;
    }
    void set(int i)
    {
        // no atomic set operation in stdc++ :(
        val = i;
    }
    void add(int i) { __atomic_add(&val,i); }
    void sub(int i) { __atomic_add(&val,-i); }
    void inc() {__atomic_add(&val,1); }
    void dec() { __atomic_add(&val,-1); }
    //bool sub_and_test_null(int i) { return __exchange_and_add(&val,-i)==i; }
    bool dec_and_test_null() { return __exchange_and_add(&val,-1)==1; }
    //bool inc_and_test_null() { return __exchange_and_add(&val,1)==-1; }
    bool add_and_test_neg(int i) { return __exchange_and_add(&val,i) < -i; }
    atomic& operator= (int i) { set(i); return *this; }
    /// Add a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator+=(int i) { add(i); }
    /// Substract a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator-=(int i) { sub(i); }
    void operator++() { inc(); }
    void operator++(int) { inc(); }
    void operator--() { dec(); }
    void operator--(int) { dec(); }

    int exchange_and_add(int i) { return __exchange_and_add(&val,i); }
    int compare_and_swap(int cmp, int with) { return __sync_val_compare_and_swap(&val, with, cmp); }

    static const char* getImplName() { return "GLIBC"; }
};

#elif defined(WIN32) || defined(_XBOX)

/// Small class used for multi-process reference counting
template<> class atomic<int>
{
    volatile LONG val;
public:
    atomic() {}
    atomic(int i) { set(i); }
    operator int() const
    {
        // this is the correct implementation
        //     return __exchange_and_add(&val,0);
        // but this is faster and should also work on x86 and ppc
        return val;
    }
    void set(int i)
    {
        // no atomic set operation in stdc++ :(
        val = i;
    }
    void add(int i) { InterlockedExchangeAdd(&val,(LONG)i); }
    void sub(int i) { InterlockedExchangeAdd(&val,(LONG)-i); }
    void inc() { InterlockedIncrement(&val); }
    void dec() { InterlockedDecrement(&val); }
    //bool sub_and_test_null(int i) { return __exchange_and_add(&val,-i)==i; }
    bool dec_and_test_null() { return InterlockedDecrement(&val)==0; }
    //bool inc_and_test_null() { return __exchange_and_add(&val,1)==-1; }
    bool add_and_test_neg(int i) { return InterlockedExchangeAdd(&val,(LONG)i) < -i; }
    atomic& operator= (int i) { set(i); return *this; }
    /// Add a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator+=(int i) { add(i);  }
    /// Substract a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator-=(int i) { sub(i); }
    void operator++() { inc(); }
    void operator++(int) { inc(); }
    void operator--() { dec(); }
    void operator--(int) { dec(); }

    int exchange_and_add(int i) { return InterlockedExchangeAdd(&val,i); }
    int compare_and_swap(int cmp, int with) { return InterlockedCompareExchange(&val, with, cmp); }
    int exchange(int i) { return InterlockedExchange(&val,i); }

    static const char* getImplName() { return "Win32"; }
};
#elif defined(PS3)
/// Small class used for multi-process reference counting
template<> class atomic<int>
{
    uint32_t val;

public:
    atomic() {}
    atomic(int i) { set(i); }
    operator int() const
    {
        // this is the correct implementation
        //     return __exchange_and_add(&val,0);
        // but this is faster and should also work on x86 and ppc
        return val;
    }
    void set(int i)
    {
		do
		{
			cellAtomicLockLine32(&val);
		}
		while (cellAtomicStoreConditional32(&val, i));
    }

	void add(int i) { cellAtomicAdd32(&val,(uint32_t)i); }
    void sub(int i) { cellAtomicSub32(&val,(uint32_t)i); }
    void inc() { cellAtomicIncr32(&val); }
    void dec() { cellAtomicDecr32(&val); }
    //bool sub_and_test_null(int i) { return __exchange_and_add(&val,-i)==i; }
    bool dec_and_test_null() { cellAtomicDecr32(&val); return val==0; }
    //bool inc_and_test_null() { return __exchange_and_add(&val,1)==-1; }
    bool add_and_test_neg(int i) { cellAtomicAdd32(&val,(uint32_t)i); return val <-i; }
    atomic& operator= (int i) { set(i); return *this; }
    /// Add a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator+=(int i) { add(i);  }
    /// Substract a value to this atomic, without return value (use exchange_and_add if you need the result)
    void operator-=(int i) { sub(i); }
    void operator++() { inc(); }
    void operator++(int) { inc(); }
    void operator--() { dec(); }
    void operator--(int) { dec(); }

    int exchange_and_add(int i) { return cellAtomicAdd32(&val,i);}
    int compare_and_swap(int cmp, int with) { return cellAtomicCompareAndSwap32(&val, with, cmp);}

	int exchange(int i)
	{
		uint32_t old;
		do
		{
			old = cellAtomicLockLine32(&val);
		}
		while (cellAtomicStoreConditional32(&val, i));

		return old;
	}

    static const char* getImplName() { return "PS3"; }
};
#endif

} // namespace system

} // namespace helper

} // namespace sofa

#endif
