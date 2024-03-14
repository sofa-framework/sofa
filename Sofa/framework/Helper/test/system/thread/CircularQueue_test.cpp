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

#include <sofa/helper/system/thread/CircularQueue.h>
#include <sofa/helper/system/thread/CircularQueue.inl>

#include <gtest/gtest.h>
#include <memory>
#include <cstdio>

using namespace sofa::helper::system::thread;

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
    ~CircularQueue_SingleTest() override
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
    ~CircularQueue_SingleProdSingleConsTest() override
    {
        waitCompletion();
    }
    void start()
    {
        producer.reset(new std::thread(std::bind(&CircularQueue_SingleProdSingleConsTest::produce, this)));
        consumer.reset(new std::thread(std::bind(&CircularQueue_SingleProdSingleConsTest::consume, this)));
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
            while(!queue.push(i))
                std::this_thread::yield();
        }
    }
    void consume()
    {
        for(int value = 0, lastValue = -1; value != 99; ++counter)
        {
            while(!queue.pop(value)) 
                std::this_thread::yield();
            EXPECT_TRUE(lastValue < value);
            lastValue = value;
        }
    }

    std::unique_ptr<std::thread> producer;
    std::unique_ptr<std::thread> consumer;
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
    ~CircularQueue_ManyProdManyConsTest() override
    {
    }
    void start()
    {
        for(int i = 0; i < ProducerCount; ++i)
        {
            prod[i].reset(new std::thread(std::bind(&CircularQueue_ManyProdManyConsTest::produce, this, i)));
        }
        for(int i = 0; i < ConsumerCount; ++i)
        {
            cons[i].reset(new std::thread(std::bind(&CircularQueue_ManyProdManyConsTest::consume, this, i)));
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
        for(int i = 0; i < TokenCount; ++i)
        {
            const int token = i + n * TokenCount;
            while(!queue.push(token));

            const int queueSize = queue.size();
            if(queueSize < 0)        emptyFault++;
            if(queueSize > Capacity) fullFault++;

        }
    }
    void consume(int n)
    {
        SOFA_UNUSED(n);

        for(std::atomic<int> value = 0; value != ExitToken; ++counter)
        {
            while(!queue.pop(value));

            const int queueSize = queue.size();
            if(queueSize < 0)        emptyFault++;
            if(queueSize > Capacity) fullFault++;
        }
    }
    std::unique_ptr<std::thread> prod[ProducerCount];
    std::unique_ptr<std::thread> cons[ConsumerCount];
    CircularQueue<std::atomic<int>, FixedPower2Size<Capacity>::type, ManyThreadsPerEnd> queue;
    std::atomic<int> counter;
    std::atomic<int> emptyFault;
    std::atomic<int> fullFault;
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

TEST_F(CircularQueue_SingleProdSingleConsTest, DISABLED_mt_1prod_1cons)
{
    start();
    waitCompletion();
    check();
}

TEST_F(CircularQueue_ManyProdManyConsTest, DISABLED_mt_3prod_3cons)
{
    start();
    waitCompletion();
    check();
}
