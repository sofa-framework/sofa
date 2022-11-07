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
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/FullVector.h>
#include <sofa/type/Vec.h>

namespace sofa::linearalgebra
{

template< std::size_t N, typename T>
class BlockVector : public FullVector<T>
{
public:
    typedef FullVector<T> Inherit;
    typedef T Real;
    typedef typename Inherit::Index Index;

    typedef typename Inherit::value_type value_type;
    typedef typename Inherit::Size Size;
    typedef typename Inherit::iterator iterator;
    typedef typename Inherit::const_iterator const_iterator;

    class Block : public sofa::type::Vec<N,T>
    {
    public:
        Index Nrows() const { return N; }
        void resize(Index) { this->clear(); }
        void operator=(const type::Vec<N,T>& v)
        {
            type::Vec<N,T>::operator=(v);
        }
        void operator=(int v)
        {
            type::Vec<N,T>::fill((float)v);
        }
        void operator=(float v)
        {
            type::Vec<N,T>::fill(v);
        }
        void operator=(double v)
        {
            type::Vec<N,T>::fill(v);
        }
    };

    typedef Block SubVectorType;

    BlockVector();

    explicit BlockVector(Index n);

    virtual ~BlockVector();

    const Block& sub(Index i, Index) const
    {
        return (const Block&)*(this->ptr()+i);
    }

    Block& sub(Index i, Index);

    const Block& asub(Index bi, Index) const;

    Block& asub(Index bi, Index);
};

#if !defined(SOFA_LINEARALGEBRA_BLOCKVECTOR_CPP)
extern template class SOFA_LINEARALGEBRA_API linearalgebra::BlockVector<6, SReal>;
#endif

} // namespace sofa::linearalgebra
