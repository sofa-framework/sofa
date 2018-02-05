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
#include <sofa/simulation/CactusStackStorage.h>

namespace sofa
{

namespace simulation
{

void CactusStackStorage::push(void* data)
{
    stack.push(data);
}
void* CactusStackStorage::pop()
{
    if (stack.empty()) return nullptr;
    void* data = stack.top();
    stack.pop();
    return data;
}
void* CactusStackStorage::top() const
{
    if (stack.empty())
        if (up)
            return up->top();
        else
            return nullptr;
    else
        return stack.top();
}
bool CactusStackStorage::empty() const
{
    return stack.empty() && (up == nullptr || up->empty());
}

} // namespace simulation

} // namespace sofa

