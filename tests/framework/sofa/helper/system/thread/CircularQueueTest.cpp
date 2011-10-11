/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Tests                                 *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/helper/system/thread/CircularQueue.h>
#include <sofa/helper/system/thread/CircularQueue.inl>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <memory>
#include <stdio.h>

using namespace sofa::helper::system::thread;
using sofa::helper::system::atomic;

struct CircularQueue_Single
{
    CircularQueue_Single()
    {
        BOOST_CHECK(queue.isEmpty());

        queue.push(1);
        queue.push(2);
        queue.push(3);
    }
    ~CircularQueue_Single()
    {
    }
    CircularQueue<int, FixedSize<6>::type, OneThreadPerEnd> queue;
};

struct SingleProdSingleCons
{
    SingleProdSingleCons()
        : counter(0)
    {
    }
    ~SingleProdSingleCons()
    {
        waitCompletion();
    }
    void start()
    {
        producer.reset(new boost::thread(boost::bind(&SingleProdSingleCons::produce, this)));
        consumer.reset(new boost::thread(boost::bind(&SingleProdSingleCons::consume, this)));
    }
    void waitCompletion()
    {
        producer->join();
        consumer->join();
    }
    void check()
    {
        BOOST_CHECK_EQUAL(counter, 100);
    }
    void produce()
    {
        for(int i = 0; i < 100; ++i)
        {
            while(!queue.push(i)) producer->yield();
        }
    }
    void consume()
    {
        for(int value = 0, lastValue = -1; value != 99; ++counter)
        {
            while(!queue.pop(value)) consumer->yield();
            BOOST_CHECK(lastValue < value);
            lastValue = value;
        }
    }

    std::auto_ptr<boost::thread> producer;
    std::auto_ptr<boost::thread> consumer;
    CircularQueue<int, FixedSize<6>::type, OneThreadPerEnd > queue;
    int counter;
};

struct ManyProdManyCons
{
    enum { Capacity = 10, ConsumerCount = 3, ProducerCount = 3, TokenCount = 100, ExitToken = 999 };

    ManyProdManyCons() : counter(0), emptyFault(0), fullFault(0)
    {
    }
    ~ManyProdManyCons()
    {
    }
    void start()
    {
        for(int i = 0; i < ProducerCount; ++i)
        {
            prod[i].reset(new boost::thread(boost::bind(&ManyProdManyCons::produce, this, i)));
        }
        for(int i = 0; i < ConsumerCount; ++i)
        {
            cons[i].reset(new boost::thread(boost::bind(&ManyProdManyCons::consume, this, i)));
        }
    }
    void waitCompletion()
    {
        for(int i = 0; i < ProducerCount; ++i) prod[i]->join();
        for(int i = 0; i < ConsumerCount; ++i) queue.push(ExitToken);
        for(int i = 0; i < ConsumerCount; ++i) cons[i]->join();
    }
    void check()
    {
        BOOST_CHECK_EQUAL(counter, ProducerCount * TokenCount + ConsumerCount);
        BOOST_CHECK_EQUAL(emptyFault, 0);
        BOOST_CHECK_EQUAL(fullFault, 0);
    }
    void produce(int n)
    {
        fprintf(stderr, "Starting producer %d\n", n); // using fprintf because it is thread-safe on Unix.

        for(int i = 0; i < TokenCount; ++i)
        {
            int token = i + n * TokenCount;
            while(!queue.push(token)) fprintf(stderr, "producer %d yield\n", n);

            int queueSize = queue.size();
            if(queueSize < 0)        emptyFault++;
            if(queueSize > Capacity) fullFault++;

            fprintf(stderr, "producer %d push %d size %d\n", n, token, queueSize);
        }
    }
    void consume(int n)
    {
        fprintf(stderr, "Starting consumer %d\n", n);

        for(atomic<int> value = 0; value != ExitToken; ++counter)
        {
            while(!queue.pop(value)) fprintf(stderr, "consumer %d yield\n", n);

            int queueSize = queue.size();
            if(queueSize < 0)        emptyFault++;
            if(queueSize > Capacity) fullFault++;

            fprintf(stderr, "consumer %d pop %d size %d\n", n, value.operator int(), queueSize);
        }
    }
    std::auto_ptr<boost::thread> prod[ProducerCount];
    std::auto_ptr<boost::thread> cons[ConsumerCount];
    CircularQueue<atomic<int>, FixedPower2Size<Capacity>::type, ManyThreadsPerEnd> queue;
    atomic<int> counter;
    atomic<int> emptyFault;
    atomic<int> fullFault;
};

BOOST_FIXTURE_TEST_CASE(pop, CircularQueue_Single)
{
    int value;
    bool result;

    result = queue.pop(value);
    BOOST_CHECK_EQUAL(result, true);
    BOOST_CHECK_EQUAL(value, 1);

    result = queue.pop(value);
    BOOST_CHECK_EQUAL(result, true);
    BOOST_CHECK_EQUAL(value, 2);

    result = queue.pop(value);
    BOOST_CHECK_EQUAL(result, true);
    BOOST_CHECK_EQUAL(value, 3);

    BOOST_CHECK(queue.isEmpty());

    value = 999;
    result = queue.pop(value);
    BOOST_CHECK_EQUAL(result, false);
    BOOST_CHECK_EQUAL(value, 999); // value not modified

    BOOST_CHECK(queue.isEmpty());
}

BOOST_FIXTURE_TEST_CASE(push, CircularQueue_Single)
{
    bool result;

    result = queue.push(4);
    BOOST_CHECK_EQUAL(result, true);

    result = queue.push(5);
    BOOST_CHECK_EQUAL(result, true);

    BOOST_CHECK(queue.isFull());

    result = queue.push(6);
    BOOST_CHECK_EQUAL(result, false);

    BOOST_CHECK(queue.isFull());
}

BOOST_FIXTURE_TEST_CASE(mt_1prod_1cons, SingleProdSingleCons)
{
    start();
    waitCompletion();
    check();
}

BOOST_FIXTURE_TEST_CASE(mt_3prod_3cons, ManyProdManyCons)
{
    start();
    waitCompletion();
    check();
}
