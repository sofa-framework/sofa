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
* RECEIVER BUFFER PART                                                        *
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
    mutex.lock();
    if (isFull())
    {
        mutex.unlock();
        throw std::out_of_range("Receiver circular buffer is full");
    }
    data[rear] = new BufferData(subject, argumentList, rows, cols);
    rear = ((this->rear + 1) % this->size);
    mutex.unlock();
}

BufferData* CircularBufferReceiver::get()
{
    mutex.lock();
    if (isEmpty())
    {
        mutex.unlock();
        throw std::out_of_range("Receiver circular buffer is empty");
    }
    BufferData* aData = this->data[front];
    front = (front + 1) % size;
    mutex.unlock();
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

CircularBufferSender::CircularBufferSender(Base* base, int size)
{
    this->base = base;
    this->data = new BaseData*[size];
    this->size = size;
}

CircularBufferSender::~CircularBufferSender()
{
    delete this->data;
}

void CircularBufferSender::add(BaseData* data)
{
    mutex.lock();
    if (isFull())
    {
        mutex.unlock();
        throw std::out_of_range("Sender circular buffer is full");
    }

    // not so proud of this part ...
    // this is the way I found for copying datas without any problem
    this->data[rear] = (data->getNewInstance());
    this->data[rear]->setHelp(data->getHelp());
    this->data[rear]->setName(data->getName());
    this->data[rear]->setOwner(this->base);
    this->data[rear]->setOwnerClass(this->base->getClassName().c_str());
    this->data[rear]->copyValue(data);

    rear = ((this->rear + 1) % this->size);
    mutex.unlock();
}

BaseData* CircularBufferSender::get()
{
    mutex.lock();
    if (isEmpty())
    {
        mutex.unlock();
        throw std::out_of_range("Sender circular buffer is empty");
    }
    BaseData* aData = this->data[front];
    front = (front + 1) % size;
    mutex.unlock();
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
