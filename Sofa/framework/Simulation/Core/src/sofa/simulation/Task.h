/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/simulation/config.h>

#include <cstddef>

namespace sofa::simulation
{
/** Task class interface    */
class SOFA_SIMULATION_CORE_API Task
{
public:
            
    /// Task Status class interface used to synchronize tasks
    class SOFA_SIMULATION_CORE_API Status
    {
    public:
        virtual ~Status() = default;
        virtual bool isBusy() const = 0;
        virtual int setBusy(bool busy) = 0;
    };
            
    /// Task Allocator class interface used to allocate tasks
    class SOFA_SIMULATION_CORE_API Allocator
    {
    public:
        virtual void* allocate(std::size_t sz) = 0;
                
        virtual void free(void* ptr, std::size_t sz) = 0;
    };
   
    Task(int scheduledThread);
            
    virtual ~Task() = default;

    enum MemoryAlloc
    {
        Stack     = 1 << 0,
        Dynamic   = 1 << 1,
        Static    = 1 << 2,
    };
            
            
    // Task interface: override these two functions
    virtual MemoryAlloc run() = 0;
            
            
    static void* operator new (std::size_t sz);
            
    // when c++14 is available delete the void  operator delete  (void* ptr)
    // and define the void operator delete  (void* ptr, std::size_t sz)
    static void  operator delete  (void* ptr);
            
    // only available in c++14.
    static void operator delete  (void* ptr, std::size_t sz);
            
    // no array new and delete operators
    static void* operator new[](std::size_t sz) = delete;
            
    // visual studio 2015 complains about the = delete but it doesn't explain where this operator is call
    // no problem with other sompilers included visual studio 2017
    // static void operator delete[](void* ptr) = delete;
            
    virtual Task::Status* getStatus(void) const = 0;
            
    int getScheduledThread() const;
            
    static Task::Allocator* getAllocator();
            
    static void setAllocator(Task::Allocator* allocator);
            
protected:

    int m_scheduledThread;
            
public:
    int m_id;
            
private:
            
    static Task::Allocator * _allocator;
};

} // namespace sofa::simulation
