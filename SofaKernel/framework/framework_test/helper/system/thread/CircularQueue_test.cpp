/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <sofa/helper/system/thread/CircularQueue.h>
#include <sofa/helper/system/thread/CircularQueue.inl>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <cstdio>

using namespace sofa::helper::system::thread;
using sofa::helper::system::atomic;

class CircularQueue_SingleTest : public ::testing::Test
{
protected:
    CircularQueue_SingleTest()
    {
        EXPECT_TRUE(queue.isEmpty());

        queue.push(1);
        queue.push(2);
        queue.push(3);
    }
    ~CircularQueue_SingleTest()
    {
    }
    CircularQueue<int, FixedSize<6>::type, OneThreadPerEnd> queue;
};

class CircularQueue_SingleProdSingleConsTest : public ::testing::Test
{
protected:
    CircularQueue_SingleProdSingleConsTest()
        : counter(0)
    {
    }
    ~CircularQueue_SingleProdSingleConsTest()
    {
        waitCompletion();
    }
    void start()
    {
        producer.reset(new boost::thread(boost::bind(&CircularQueue_SingleProdSingleConsTest::produce, this)));
        consumer.reset(new boost::thread(boost::bind(&CircularQueue_SingleProdSingleConsTest::consume, this)));
    }
    void waitCompletion()
    {
        producer->join();
        consumer->join();
    }
    void check()
    {
        EXPECT_EQ(counter, 100);
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
            EXPECT_TRUE(lastValue < value);
            lastValue = value;
        }
    }

    std::auto_ptr<boost::thread> producer;
    std::auto_ptr<boost::thread> consumer;
    CircularQueue<int, FixedSize<6>::type, OneThreadPerEnd > queue;
    int counter;
};

class CircularQueue_ManyProdManyConsTest : public ::testing::Test
{
protected:
    enum { Capacity = 10, ConsumerCount = 3, ProducerCount = 3, TokenCount = 100, ExitToken = 999 };

    CircularQueue_ManyProdManyConsTest() : counter(0), emptyFault(0), fullFault(0)
    {
    }
    ~CircularQueue_ManyProdManyConsTest()
    {
    }
    void start()
    {
        for(int i = 0; i < ProducerCount; ++i)
        {
            prod[i].reset(new boost::thread(boost::bind(&CircularQueue_ManyProdManyConsTest::produce, this, i)));
        }
        for(int i = 0; i < ConsumerCount; ++i)
        {
            cons[i].reset(new boost::thread(boost::bind(&CircularQueue_ManyProdManyConsTest::consume, this, i)));
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
        EXPECT_EQ(counter, ProducerCount * TokenCount + ConsumerCount);
        EXPECT_EQ(emptyFault, 0);
        EXPECT_EQ(fullFault, 0);
    }
    void produce(int n)
    {
        // fprintf(stderr, "Starting producer %d\n", n); // using fprintf because it is thread-safe on Unix.

        for(int i = 0; i < TokenCount; ++i)
        {
            int token = i + n * TokenCount;
            while(!queue.push(token));// fprintf(stderr, "producer %d yield\n", n);

            int queueSize = queue.size();
            if(queueSize < 0)        emptyFault++;
            if(queueSize > Capacity) fullFault++;

            // fprintf(stderr, "producer %d push %d size %d\n", n, token, queueSize);
        }
    }
    void consume(int n)
    {
        // fprintf(stderr, "Starting consumer %d\n", n);

        for(atomic<int> value = 0; value != ExitToken; ++counter)
        {
            while(!queue.pop(value));// fprintf(stderr, "consumer %d yield\n", n);

            int queueSize = queue.size();
            if(queueSize < 0)        emptyFault++;
            if(queueSize > Capacity) fullFault++;

            // fprintf(stderr, "consumer %d pop %d size %d\n", n, value.operator int(), queueSize);
        }
    }
    std::auto_ptr<boost::thread> prod[ProducerCount];
    std::auto_ptr<boost::thread> cons[ConsumerCount];
    CircularQueue<atomic<int>, FixedPower2Size<Capacity>::type, ManyThreadsPerEnd> queue;
    atomic<int> counter;
    atomic<int> emptyFault;
    atomic<int> fullFault;
};

TEST_F(CircularQueue_SingleTest, pop)
{
    int value;
    bool result;

    result = queue.pop(value);
    EXPECT_EQ(result, true);
    EXPECT_EQ(value, 1);

    result = queue.pop(value);
    EXPECT_EQ(result, true);
    EXPECT_EQ(value, 2);

    result = queue.pop(value);
    EXPECT_EQ(result, true);
    EXPECT_EQ(value, 3);

    EXPECT_TRUE(queue.isEmpty());

    value = 999;
    result = queue.pop(value);
    EXPECT_EQ(result, false);
    EXPECT_EQ(value, 999); // value not modified

    EXPECT_TRUE(queue.isEmpty());
}

TEST_F(CircularQueue_SingleTest, push)
{
    bool result;

    result = queue.push(4);
    EXPECT_EQ(result, true);

    result = queue.push(5);
    EXPECT_EQ(result, true);

    EXPECT_TRUE(queue.isFull());

    result = queue.push(6);
    EXPECT_EQ(result, false);

    EXPECT_TRUE(queue.isFull());
}

TEST_F(CircularQueue_SingleProdSingleConsTest, mt_1prod_1cons)
{
    start();
    waitCompletion();
    check();
}

TEST_F(CircularQueue_ManyProdManyConsTest, mt_3prod_3cons)
{
    start();
    waitCompletion();
    check();
}
