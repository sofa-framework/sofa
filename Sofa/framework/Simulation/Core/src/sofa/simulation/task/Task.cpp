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
#include <sofa/simulation/task/Task.h>

namespace sofa::simulation
{
Task::Allocator* Task::_allocator = nullptr;

Task::Task(int scheduledThread)
: m_scheduledThread(scheduledThread)
, m_id(0)
{
}

void *Task::operator new(std::size_t sz)
{
    return _allocator->allocate(sz);
}

void Task::operator delete(void *ptr)
{
    _allocator->free(ptr, 0);
}

void Task::operator delete(void *ptr, std::size_t sz)
{
    _allocator->free(ptr, sz);
}

int Task::getScheduledThread() const
{
    return m_scheduledThread;
}

Task::Allocator *Task::getAllocator()
{
    return _allocator;
}

void Task::setAllocator(Task::Allocator *allocator)
{
    _allocator = allocator;
}

} // namespace sofa
