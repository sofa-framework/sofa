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

#include <set>
#include <sofa/core/config.h>

namespace sofa::core
{

struct CachedDataObserver;


/**
 * Observable object notifying as soon as there is reason to change a cached data.
 *
 * An example of use case is to precompute a costly computation and cache the result
 * for faster requests.
 */
struct SOFA_CORE_API CachedDataObservable
{
    friend struct CachedDataObserver;

    ~CachedDataObservable();

    void subscribe(CachedDataObserver* observer);
    void unsubscribe(CachedDataObserver* observer);

protected:

    std::set<CachedDataObserver*> m_observers;

    /**
     * Function to call when the cache is invalid
     */
    void notifyObservers() const;
};

}
