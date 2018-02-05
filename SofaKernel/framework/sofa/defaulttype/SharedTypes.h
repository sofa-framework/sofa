/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_SHAREDTYPES_H
#define SOFA_SHAREDTYPES_H
#ifdef SOFA_SMP

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/accessor.h>
#include <iostream>
#include <algorithm>
#include <AthapascanIterative.h>
#include <Shared.h>

//using namespace should be removed from the header file
using namespace Iterative;
using a1::Shared_r;
using a1::Shared_rw;
using a1::Shared_cw;
using a1::Shared_w;
using Iterative::Shared;

namespace sofa
{

namespace defaulttype
{
namespace SharedTypes=Iterative;


template<class T>
struct sparseCumul
{
    void operator()(T& result, const T& value)
    {
        if (!value.index)
        {
            result+=value;
            return;
        }
        helper::vector<unsigned int>::iterator i=value.index->begin();
        helper::vector<unsigned int>::iterator iend=value.index->end();
        for (; i!=iend; i++)
        {

            result[*i]+=value[*i];
        }

    }
};

template<class T>
Shared< Data<T> >* getShared(const Data<T> &data)
{
    if (data.shared == NULL)
    {
        data.shared = new Shared< Data<T> >((Data<T>*) &data);
    }
    return (Shared< Data<T> >*)&data.shared;
}

template< class T, class MemoryManager = helper::CPUMemoryManager<T> >
class SharedVector: public helper::vector<T,MemoryManager>
{
public:
    typedef helper::CPUMemoryManager<T> Alloc;
    typedef typename helper::vector<T,Alloc>::size_type size_type;
    /// reference to a value (read-write)
    typedef typename helper::vector<T,Alloc>::reference reference;
    /// const reference to a value (read only)
    typedef typename helper::vector<T,Alloc>::const_reference const_reference;
    typedef sparseCumul<vector<T> > CumulOperator;
    helper::vector<unsigned int > *index;
    Shared<vector<T,Alloc> > * sharedData;
    /// Basic onstructor


    SharedVector() : helper::vector<T,Alloc>(),index(NULL),sharedData(new Shared<vector<T,Alloc> >(this)) {}
    /// Constructor
    SharedVector(size_type n, const T& value):helper::vector<T,Alloc>(n,value),index(NULL),sharedData(new Shared<vector<T,Alloc> >(this)) {}
    /// Constructor
    SharedVector(int n, const T& value): helper::vector<T,Alloc>(n,value),index(NULL) ,sharedData(new Shared<vector<T,Alloc> >(this)) {}
    /// Constructor
    SharedVector(long n, const T& value): helper::vector<T,Alloc>(n,value),index(NULL) ,sharedData(new Shared<vector<T,Alloc> >(this)) {}
    /// Constructor
    explicit SharedVector(size_type n): helper::vector<T,Alloc>(n),index(NULL),sharedData(new Shared<vector<T,Alloc> >(this)) {}
    /// Constructor
    SharedVector(const helper::vector<T, Alloc>& x): helper::vector<T,Alloc>(x),index(NULL),sharedData(new Shared<vector<T,Alloc> >(this)) {}
    /// Constructor
    SharedVector(const SharedVector<T,Alloc> &cp): helper::vector<T,Alloc>(cp),index(NULL),sharedData(new Shared<vector<T,Alloc> >(this)) {}
    SharedVector<T,Alloc> &operator=(const SharedVector<T,Alloc> &cp)
    {
        helper::vector<T,Alloc>::operator=(cp);
        return *this;
    }
    ~SharedVector()
    {
        if (index) delete index;
    }

    a1::Shared<vector<T,Alloc> >& operator*() const
    {
        return *sharedData;
    }
    SharedVector<T, Alloc>& operator+=(const helper::vector<T, Alloc>& x)
    {
        // do stuff.
        for (unsigned int i=0; i<x.size()&&i<this->size(); i++)
            (*this)[i]+=x[i];
        return *this;
    }
    void zero()
    {
        if (!index)
        {
            index=new helper::vector<unsigned int >();

        }
        for (unsigned int i=0; i<index->size(); i++)
            (*this)[(*index)[i]]=T();
        index->clear();

    }

};


}
namespace helper
{
using namespace defaulttype;
template<class T, class Alloc>
class ReadAccessor< SharedVector<T,Alloc> > : public ReadAccessorVector< SharedVector<T,Alloc> >
{
public:
    typedef ReadAccessorVector< SharedVector<T,Alloc> > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class T, class Alloc>
class WriteAccessor< SharedVector<T,Alloc> > : public WriteAccessorVector< SharedVector<T,Alloc> >
{
public:
    typedef WriteAccessorVector< SharedVector<T,Alloc> > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};
}
} // namespace sofa


template<typename T>
a1::IStream& operator >> ( a1::IStream& in, T&  )
{
    std::cerr<<"Communicator not implemented!!"<<std::endl;
    //in>>v;
    return in;
}
template<typename T>
a1::OStream& operator << ( a1::OStream& out, const T&  )
{

    std::cerr<<"Communicator not implemented!!"<<std::endl;


    return out;
}

#endif
#endif
