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
#include <sofa/core/CachedDataObservable.h>
#include <sofa/core/CachedDataObserver.h>
#include <gtest/gtest.h>

namespace sofa
{

TEST(CachedDataObservable, regular)
{
    core::CachedDataObservable observable;
    core::CachedDataObserver observer;
    observable.subscribe(&observer);

    EXPECT_FALSE(observer.isCacheValid());

    observer.invalidCache();
    EXPECT_FALSE(observer.isCacheValid());

    observer.setCacheValid();
    EXPECT_TRUE(observer.isCacheValid());

    observer.invalidCache();
    EXPECT_FALSE(observer.isCacheValid());
}

TEST(CachedDataObservable, observerDestroyedBeforeObservable)
{
    struct CountObservers : core::CachedDataObservable
    {
        [[nodiscard]] std::size_t getNumberOfObservers() const
        {
            return m_observers.size();
        }
    } observable;

    EXPECT_EQ(observable.getNumberOfObservers(), 0);
    {
        core::CachedDataObserver observer;
        observable.subscribe(&observer);

        EXPECT_EQ(observable.getNumberOfObservers(), 1);
    }

    EXPECT_EQ(observable.getNumberOfObservers(), 0);
}

TEST(CachedDataObservable, observableDestroyedBeforeObserver)
{
    auto observable = std::make_shared<core::CachedDataObservable>();

    struct CountObservable : core::CachedDataObserver
    {
        void postObservableDestroyed(core::CachedDataObservable* observable) override
        {
            hasBeenNotified = true;
        }

        bool hasBeenNotified { false };
    } observer;
    observable->subscribe(&observer);

    EXPECT_FALSE(observer.hasBeenNotified);

    observable.reset();

    EXPECT_TRUE(observer.hasBeenNotified);

}


}
