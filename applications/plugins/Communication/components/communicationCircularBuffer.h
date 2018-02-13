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
#ifndef SOFA_COMMUNICATIONCIRCULARBUFFER_H
#define SOFA_COMMUNICATIONCIRCULARBUFFER_H

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::Base;

namespace sofa
{

namespace component
{

namespace communication
{

typedef std::vector<std::string> ArgumentList;

static std::mutex mutex;

class BufferData
{
public:
    BufferData(){}
    BufferData(std::string subject, ArgumentList argumentList, int rows, int cols)
    {
        this->subject = subject;
        this->argumentList = argumentList;
        this->rows = rows;
        this->cols = cols;
    }

    int getRows() const;
    int getCols() const;
    ArgumentList getArgumentList() const;
    std::string getSubject() const;

private:
    std::string subject;
    ArgumentList argumentList;
    int rows ;
    int cols ;
};

class CircularBufferReceiver
{
public:

    CircularBufferReceiver(int size);
    ~CircularBufferReceiver();

    void add(std::string subject, ArgumentList argumentList, int rows, int cols);
    BufferData* get();

private:

    int front = 0;
    int rear = 0;
    BufferData ** data;
    int size;

    bool isEmpty();
    bool isFull();

};

class CircularBufferSender
{
public:

    CircularBufferSender(Base* base, int size);
    ~CircularBufferSender();

    void add(BaseData* data);
    BaseData* get();

private:

    Base* base;
    int front = 0;
    int rear = 0;
    BaseData ** data;
    int size;

    bool isEmpty();
    bool isFull();

};

} /// namespace communication
} /// namespace component
} /// namespace sofa

#endif // SOFA_COMMUNICATIONCIRCULARBUFFER_H
