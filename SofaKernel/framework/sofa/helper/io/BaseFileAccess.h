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

#ifndef SOFA_HELPER_IO_BASEFILEACCESS_H
#define SOFA_HELPER_IO_BASEFILEACCESS_H

#include <sofa/helper/helper.h>

#include <iostream>
#include <string>

namespace sofa
{

namespace helper
{

namespace io
{

class BaseFileAccess;

class BaseFileAccessCreator
{
public:
    virtual ~BaseFileAccessCreator() {}

    virtual BaseFileAccess* create() const = 0;

};

template<class T>
class FileAccessCreator : public BaseFileAccessCreator
{
public:
    virtual T* create() const
    {
        return new T();
    }
};

// \brief The goal of this class is to unify the way we access files in Sofa and to be able to change this way, transparently, according to the need of the end-user application
class SOFA_HELPER_API BaseFileAccess
{
public:
    static void SetDefaultCreator(); // \warning: Should be called only from the end-user application to avoid undesired FileAccess overriding
    static void SetCreator(BaseFileAccessCreator* baseFileAccessCreator); // \warning: Should be called only from the end-user application to avoid undesired FileAccess overriding
    template<class T>
    static void SetCreator(); // \warning: Should be called only from the end-user application to avoid undesired FileAccess overriding
    static BaseFileAccess* Create();

protected:
    BaseFileAccess();

public:
    virtual ~BaseFileAccess();

    virtual bool open(const std::string& filename, std::ios_base::openmode openMode) = 0;
    virtual void close() = 0;

    virtual std::streambuf* streambuf() const = 0;
    virtual std::string readAll() = 0;
    virtual void write(const std::string& data) = 0;

private:
    static BaseFileAccessCreator* OurCreator;

};

template<class T>
inline void BaseFileAccess::SetCreator()
{
    delete OurCreator;
    OurCreator = new FileAccessCreator<T>();
}

} // namespace io

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_IO_BASEFILEACCESS_H
