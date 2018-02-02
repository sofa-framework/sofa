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
#include <Communication/components/communicationCircularBuffer.h>

namespace sofa
{

namespace component
{

namespace communication
{


int BufferData::getRows() const
{
    return rows;
}

int BufferData::getCols() const
{
    return cols;
}

ArgumentList BufferData::getArgumentList() const
{
    return argumentList;
}

std::string BufferData::getSubject() const
{
    return subject;
}

/******************************************************************************
*                                                                             *
* SENDER BUFFER PART                                                          *
*                                                                             *
******************************************************************************/

CircularBufferReceiver::CircularBufferReceiver(int size)
{
    this->data = new BufferData*[size];
    this->size = size;
}

CircularBufferReceiver::~CircularBufferReceiver()
{
    delete this->data;
}

void CircularBufferReceiver::add(std::string subject, ArgumentList argumentList, int rows, int cols)
{
    pthread_mutex_lock(&mutex);
    if (isFull())
    {
        pthread_mutex_unlock(&mutex);
        throw std::out_of_range("Circular buffer is full");
    }
    data[rear] = new BufferData(subject, argumentList, rows, cols);
    rear = ((this->rear + 1) % this->size);
    pthread_mutex_unlock(&mutex);
}


BufferData* CircularBufferReceiver::get()
{
    pthread_mutex_lock(&mutex);
    if (isEmpty())
    {
        pthread_mutex_unlock(&mutex);
        throw std::out_of_range("Circular buffer is empty");
    }
    BufferData* aData = this->data[front];
    front = (front + 1) % size;
    pthread_mutex_unlock(&mutex);
    return aData;
}

bool CircularBufferReceiver::isEmpty()
{
    return rear == front;
}

bool CircularBufferReceiver::isFull()
{
    return ((this->rear + 1) % this->size) == front;
}

/******************************************************************************
*                                                                             *
* SENDER BUFFER PART                                                          *
*                                                                             *
******************************************************************************/

CircularBufferSender::CircularBufferSender(int size)
{
    this->data = new BaseData*[size];
    this->size = size;
}

CircularBufferSender::~CircularBufferSender()
{
//    delete this->data;
}

void CircularBufferSender::add(BaseData* data)
{
    pthread_mutex_lock(&mutex);
    if (isFull())
    {
        pthread_mutex_unlock(&mutex);
        throw std::out_of_range("Circular buffer is full");
    }
    this->data[rear] = (data->clone());
    rear = ((this->rear + 1) % this->size);

    pthread_mutex_unlock(&mutex);
}

BaseData* CircularBufferSender::get()
{
    pthread_mutex_lock(&mutex);
    if (isEmpty())
    {
        pthread_mutex_unlock(&mutex);
        throw std::out_of_range("Circular buffer is empty");
    }
    BaseData* aData = this->data[front];
    front = (front + 1) % size;
    pthread_mutex_unlock(&mutex);
    return aData;
}

bool CircularBufferSender::isEmpty()
{
    return rear == front;
}

bool CircularBufferSender::isFull()
{
    return ((this->rear + 1) % this->size) == front;
}

} /// communication

} /// component

} /// sofa
