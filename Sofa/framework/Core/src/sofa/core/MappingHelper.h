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
#include <sofa/core/config.h>

#include <sofa/type/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::core
{

    template<class T1, class T2>
    static inline void eq(T1& dest, const T2& src)
    {
        dest = src;
    }

    template<class T1, class T2>
    static inline void peq(T1& dest, const T2& src)
    {
        dest += src;
    }

    // float <-> double (to remove warnings)

    //template<>
    static inline void eq(float& dest, const double& src)
    {
        dest = (float)src;
    }

    //template<>
    static inline void peq(float& dest, const double& src)
    {
        dest += (float)src;
    }

    // Vec <-> Vec

    template<Size N1, Size N2, class T1, class T2>
    static inline void eq(type::Vec<N1,T1>& dest, const type::Vec<N2,T2>& src)
    {
        dest = type::toVecN<N1, T1>(src);
    }

    template<Size N1, Size N2, class T1, class T2>
    static inline void peq(type::Vec<N1,T1>& dest, const type::Vec<N2,T2>& src)
    {
        for (Size i=0; i<(N1>N2?N2:N1); i++)
            dest[i] += (T1)src[i];
    }

    // RigidDeriv <-> RigidDeriv

    template<Size N, class T1, class T2>
    static inline void eq(defaulttype::RigidDeriv<N,T1>& dest, const defaulttype::RigidDeriv<N,T2>& src)
    {
        dest.getVCenter() = src.getVCenter();
        dest.getVOrientation() = (typename defaulttype::RigidDeriv<N,T1>::Rot)src.getVOrientation();
    }

    template<Size N, class T1, class T2>
    static inline void peq(defaulttype::RigidDeriv<N,T1>& dest, const defaulttype::RigidDeriv<N,T2>& src)
    {
        dest.getVCenter() += src.getVCenter();
        dest.getVOrientation() += (typename defaulttype::RigidDeriv<N,T1>::Rot)src.getVOrientation();
    }

    // RigidCoord <-> RigidCoord

    template<Size N, class T1, class T2>
    static inline void eq(defaulttype::RigidCoord<N,T1>& dest, const defaulttype::RigidCoord<N,T2>& src)
    {
        dest.getCenter() = src.getCenter();
        dest.getOrientation() = (typename defaulttype::RigidCoord<N,T1>::Rot)src.getOrientation();
    }

    template<Size N, class T1, class T2>
    static inline void peq(defaulttype::RigidCoord<N,T1>& dest, const defaulttype::RigidCoord<N,T2>& src)
    {
        dest.getCenter() += src.getCenter();
        dest.getOrientation() += src.getOrientation();
    }

    // RigidDeriv <-> Vec

    template<Size N, class T1, class T2>
    static inline void eq(type::Vec<N,T1>& dest, const defaulttype::RigidDeriv<N,T2>& src)
    {
        dest = src.getVCenter();
    }

    template<Size N, class T1, class T2>
    static inline void peq(type::Vec<N,T1>& dest, const defaulttype::RigidDeriv<N,T2>& src)
    {
        dest += src.getVCenter();
    }

    template<Size N, class T1, class T2>
    static inline void eq(defaulttype::RigidDeriv<N,T1>& dest, const type::Vec<N,T2>& src)
    {
        dest.getVCenter() = src;
    }

    template<Size N, class T1, class T2>
    static inline void peq(defaulttype::RigidDeriv<N,T1>& dest, const type::Vec<N,T2>& src)
    {
        dest.getVCenter() += src;
    }

    // RigidCoord <-> Vec
    template<Size N, class T1, class T2>
    static inline void eq(type::Vec<N,T1>& dest, const defaulttype::RigidCoord<N,T2>& src)
    {
        dest = src.getCenter();
    }

    template<Size N, class T1, class T2>
    static inline void peq(type::Vec<N,T1>& dest, const defaulttype::RigidCoord<N,T2>& src)
    {
        dest += src.getCenter();
    }

    template<Size N, class T1, class T2>
    static inline void eq(defaulttype::RigidCoord<N,T1>& dest, const type::Vec<N,T2>& src)
    {
        dest.getCenter() = src;
    }

    template<Size N, class T1, class T2>
    static inline void peq(defaulttype::RigidCoord<N,T1>& dest, const type::Vec<N,T2>& src)
    {
        dest.getCenter() += src;
    }

} // namespace sofa::component::mapping
