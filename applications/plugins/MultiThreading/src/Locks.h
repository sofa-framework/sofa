/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef MultiThreadingLocks_h__
#define MultiThreadingLocks_h__

#include <MultiThreading/config.h>

#include <thread>
#include <atomic>

namespace sofa
{

	namespace simulation
	{


        class SpinLock
        {
            enum
            {
                CACHE_LINE = 64
            };
            
        public:
            
            SpinLock()
            {}
            
            ~SpinLock()
            {
                unlock();
            }
            
            bool try_lock()
            {
                return !_flag.test_and_set( std::memory_order_acquire );
            }
            
            void lock()
            {
                while( _flag.test_and_set(std::memory_order_acquire) )
                {
                    // cpu busy wait
                    std::this_thread::yield();
                }
            }
            
            void unlock()
            {
                _flag.clear( std::memory_order_release );
            }
            
        private:
            
            std::atomic_flag _flag = ATOMIC_FLAG_INIT;
            
            char _pad [CACHE_LINE - sizeof(std::atomic_flag)];
        };
        
        
        
        class ScopedLock
        {
        public:
            
            explicit ScopedLock( SpinLock & lock ): _spinlock( lock )
            {
                _spinlock.lock();
            }
            
            ~ScopedLock()
            {
                _spinlock.unlock();
            }
            
            ScopedLock( ScopedLock const & ) = delete;
            ScopedLock & operator=( ScopedLock const & ) = delete;
            
        private:
            
            SpinLock& _spinlock;
        };

	} // namespace simulation

} // namespace sofa


#endif // MultiThreadingLocks_h__
