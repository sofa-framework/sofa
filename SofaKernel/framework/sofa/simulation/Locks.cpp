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
#include <sofa/simulation/Locks.h>


namespace sofa
{

	namespace simulation
	{



        SpinLock::SpinLock()
        : _flag()
        {
            
        }
            

        SpinLock::~SpinLock()
        {
            unlock();
        }
        
        bool SpinLock::try_lock()
        {
            return !_flag.test_and_set( std::memory_order_acquire );
        }
        
        void SpinLock::lock()
        {
            while( _flag.test_and_set(std::memory_order_acquire) )
            {
                // cpu busy wait
                //std::this_thread::yield();
            }
        }
        
        void SpinLock::unlock()
        {
            _flag.clear( std::memory_order_release );
        }
        


	} // namespace simulation

} // namespace sofa
