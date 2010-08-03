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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL

#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace mapping
{

template<class T1, class T2>
extern void eq(T1& dest, const T2& src)
{
    dest = (T1)(src);
}

template<class T1, class T2>
extern void peq(T1& dest, const T2& src)
{
    dest += (T1)(src);
}

template<int N, typename real1, typename real2>
extern void eq(defaulttype::RigidCoord<N,real1>& dest, const defaulttype::RigidCoord<N,real2>& src)
{
    dest.getCenter() = src.getCenter();
    dest.getOrientation() = src.getOrientation();
}

template<typename real1, typename real2>
extern void eq(defaulttype::RigidCoord<2,real1>& dest, const defaulttype::RigidCoord<2,real2>& src)
{
    dest.getCenter() = src.getCenter();
    dest.getOrientation() = (real1)src.getOrientation();
}

template<int N, typename real1, typename real2>
extern void peq(defaulttype::RigidCoord<N,real1>& dest, const defaulttype::RigidCoord<N,real2>& src)
{
    dest.getCenter() += src.getCenter();
    dest.getOrientation() += src.getOrientation();
}

template<typename real1, typename real2>
extern void peq(defaulttype::RigidCoord<2,real1>& dest, const defaulttype::RigidCoord<2,real2>& src)
{
    dest.getCenter() += src.getCenter();
    dest.getOrientation() += (real1)src.getOrientation();
}

template<int N, typename real1, typename real2>
extern void eq(defaulttype::RigidDeriv<N,real1>& dest, const defaulttype::RigidDeriv<N,real2>& src)
{
    dest.getVCenter() = src.getVCenter();
    dest.getVOrientation() = src.getVOrientation();
}

template<typename real1, typename real2>
extern void eq(defaulttype::RigidDeriv<2,real1>& dest, const defaulttype::RigidDeriv<2,real2>& src)
{
    dest.getVCenter() = src.getVCenter();
    dest.getVOrientation() = (real1)src.getVOrientation();
}

template<int N, typename real1, typename real2>
extern void peq(defaulttype::RigidDeriv<N,real1>& dest, const defaulttype::RigidDeriv<N,real2>& src)
{
    dest.getVCenter() += src.getVCenter();
    dest.getVOrientation() += src.getVOrientation();
}

template<typename real1, typename real2>
extern void peq(defaulttype::RigidDeriv<2,real1>& dest, const defaulttype::RigidDeriv<2,real2>& src)
{
    dest.getVCenter() += src.getVCenter();
    dest.getVOrientation() += (real1)src.getVOrientation();
}

template<int N, typename real1, typename real2>
extern void eq(defaulttype::Vec<N,real1>& dest, const defaulttype::RigidCoord<N,real2>& src)
{
    dest = src.getCenter();
}

template<int N, typename real1, typename real2>
extern void peq(defaulttype::Vec<N,real1>& dest, const defaulttype::RigidCoord<N,real2>& src)
{
    dest += src.getCenter();
}

template<int N, typename real1, typename real2>
extern void eq(defaulttype::Vec<N,real1>& dest, const defaulttype::RigidDeriv<N,real2>& src)
{
    dest = src.getVCenter();
}

template<int N, typename real1, typename real2>
extern void peq(defaulttype::Vec<N,real1>& dest, const defaulttype::RigidDeriv<N,real2>& src)
{
    dest += src.getVCenter();
}

template<int N, typename real1, typename real2>
extern void eq(defaulttype::RigidCoord<N,real1>& dest, const defaulttype::Vec<N,real2>& src)
{
    dest.getCenter() = src;
}

template<int N, typename real1, typename real2>
extern void peq(defaulttype::RigidCoord<N,real1>& dest, const defaulttype::Vec<N,real2>& src)
{
    dest.getCenter() += src;
}

template<int N, typename real1, typename real2>
extern void eq(defaulttype::RigidDeriv<N,real1>& dest, const defaulttype::Vec<N,real2>& src)
{
    dest.getVCenter() = src;
}

template<int N, typename real1, typename real2>
extern void peq(defaulttype::RigidDeriv<N,real1>& dest, const defaulttype::Vec<N,real2>& src)
{
    dest.getVCenter() += src;
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{


    //const unsigned int N = Coord::size() < InCoord::size() ? Coord::size() : InCoord::size();
    out.resize(in.size());
    for(unsigned int i=0; i<out.size(); i++)
    {
        //for (unsigned int j=0;j < N;++j)
        //    out[i][j] = (OutReal)in[i][j];
        //out[i] = in[i];
        eq(out[i], in[i]);
    }
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{


    //const unsigned int N = Deriv::size() < InDeriv::size() ? Deriv::size() : InDeriv::size();
    out.resize(in.size());

    if ( !(maskTo->isInUse()) )
    {
        for(unsigned int i=0; i<out.size(); i++)
        {
            //for (unsigned int j=0;j < N;++j)
            //    out[i][j] = (OutReal)in[i][j];
            eq(out[i], in[i]);
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();
        ParticleMask::InternalStorage::const_iterator it;
        for (it=indices.begin(); it!=indices.end(); it++)
        {
            const int i=(int)(*it);
            eq(out[i], in[i]);
        }
    }
}

template <class BasicMapping>
void IdentityMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    //const unsigned int N = Deriv::size() < InDeriv::size() ? Deriv::size() : InDeriv::size();

    if ( !(maskTo->isInUse()) )
    {
        maskFrom->setInUse(false);
        for(unsigned int i=0; i<in.size(); i++)
        {
            //for (unsigned int j=0;j < N;++j)
            //    out[i][j] += (Real)in[i][j];
            peq(out[i], in[i]);
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();
        ParticleMask::InternalStorage::const_iterator it;
        for (it=indices.begin(); it!=indices.end(); it++)
        {
            const int i=(int)(*it);
            peq(out[i], in[i]);
            maskFrom->insertEntry(i);
        }
    }
}

template <class BaseMapping>
void IdentityMapping<BaseMapping>::applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                InDeriv data;
                eq(data, colIt.val());

                o.addCol(colIt.index(), data);

                ++colIt;
            }
        }
    }
}

template <class BaseMapping>
void IdentityMapping<BaseMapping>::handleTopologyChange()
{
    if ( stateTo && stateFrom && stateTo->getSize() != stateFrom->getSize()) this->init();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
