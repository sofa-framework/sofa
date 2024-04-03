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

struct CachedDataObservable;

struct SOFA_CORE_API CachedDataObserver
{
    friend struct CachedDataObservable;

    virtual ~CachedDataObserver();

    void invalidCache();
    [[nodiscard]] bool isCacheValid() const;
    void setCacheValid();

protected:
    bool m_isCacheValid { false };

    /**
     * Notify when the observable is destroyed
     */
    void notifyObservableDestroyed(CachedDataObservable* observable);

    virtual void postObservableDestroyed(CachedDataObservable* observable);

private:
    std::set<CachedDataObservable*> m_observables;
};

}
