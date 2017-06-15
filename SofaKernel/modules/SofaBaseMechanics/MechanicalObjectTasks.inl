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
#ifndef SOFA_COMPONENT_PARALLELMECHANICALOBJECTTASKS_H
#define SOFA_COMPONENT_PARALLELMECHANICALOBJECTTASKS_H

#include <sofa/helper/accessor.h>
#include <athapascan-1>

namespace sofa
{

namespace component
{
using namespace a1;
struct cumulD
{
    void operator () (double &res, const double &val)
    {
        res += val;
    }

};

template < class T > class resizeVec
{
public:
    void operator   () (void*,Shared_rw < T > v, int size)
    {
        helper::WriteAccessor<T> vv = v.access ();
        vv.resize (size);
    }
};
template < class T > class printDOFSh
{
public:
    void operator     () (void*,Shared_r < T > _v)
    {
        helper::ReadAccessor<T> v = _v.read ();
        if (v.size ())
        {
            std::cerr << "pDOF:" << v << std::endl;
        }
    }
};

template < class T > class VecInit
{
public:
    void operator     () (void*,Shared_rw < T > vv)
    {
#if 1
        helper::WriteAccessor<T> v = vv.access ();
        v.clear ();
#else
        vv.write (T ());
#endif
    }
};
template < class T > class VecInitResize
{
public:
    void operator     () (void*,Shared_rw < T > vv, unsigned int vsize)
    {
#if 1
        helper::WriteAccessor<T> v = vv.access ();
        v.clear ();
        v.resize (vsize);
#else

        T v;
        //      =vv.access();
        v.resize (vsize);

        vv.write (v);
#endif
    }
};

template < class T, class T2 > class vClear
{
public:
    void operator     () (void *,Shared_rw < T > vv)
    {

        helper::WriteAccessor<T> vc = vv.access ();
        for (unsigned int i = 0; i < vc.size (); i++)
            (vc)[i] = T2 ();
    }
    void operator     () (void*,Shared_rw < T > vv, unsigned vsize)
    {

        helper::WriteAccessor<T> vc1 = vv.access ();
        vc1.resize (vsize);
        helper::WriteAccessor<T> vc = vv.access ();
        for (unsigned int i = 0; i < vc.size (); i++)
            (vc)[i] = T2 ();
    }
};

template < class T , class T2> class vTEq
{
public:
    void operator     () (void*,Shared_rw <T >
            vv, double f)
    {
        helper::WriteAccessor<T> vd = vv.access ();

        for (unsigned int i = 0; i < vd.size (); i++)
            (vd)[i] *= ( T2) f;
    }
};







template < class TVec,class TReal > class vEqBF
{
public:
    void operator     () (void*,Shared_rw < TVec >
            vv,
            Shared_r < TVec > vb,
            double f)
    {
        helper::WriteAccessor<TVec> vv2 = vv.access ();
        helper::ReadAccessor<TVec> vb2 = vb.read ();

        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            (vv2)[i] = vb2[i] * (TReal) f;
        }
    }
};





template < class TVec > class vAssign
{
public:
    void operator     () (void*,Shared_rw < TVec > vv, Shared_r < TVec > vb)
    {
        helper::WriteAccessor<TVec> vv2 = vv.access ();
        helper::ReadAccessor<TVec> vb2 = vb.read ();

        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
            (vv2)[i] = vb2[i];
    }
};
template < class T1, class T2 > class vPEq
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();
        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            (vv2)[i] += vb2[i];
        }
    }
};
template <  class T1, class T2 > class vPEq2
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb)
    {
        helper::ReadAccessor<T2> vb2 = vb.read ();
        if (vb2.size()==0) return;
        helper::WriteAccessor<T1> vv2 = vv.access ();
        for (unsigned int i = 0; i < vb2.size (); i++)
        {
            (vv2)[i] += vb2[i];
        }
    }
};
template < class DataTypes, class T1, class T2 > class vPEqBF
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb,
            Shared_r < double >_f, typename DataTypes::Real)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        const typename DataTypes::Real & f = _f.read ();
        helper::ReadAccessor<T2> vb2 = vb.read ();

        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            (vv2)[i] += vb2[i] * f;
        }

    }
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb,
            typename DataTypes::Real f)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();

        vv2.resize (vb2.size ());

        for (unsigned int i = 0; i < vv2.size (); i++)
            (vv2)[i] += vb2[i] * f;

    }
};

/// NOTE : DataTypes is useless
template < class DataTypes, class T1 > class vDotOp
{
public:
    void operator     () (void*,Shared_r < T1 > _va, Shared_rw < double >res)
    {
        helper::ReadAccessor<T1> va = _va.read ();

        //double tmp=0;
        double &tmp = res.access ();

        for (unsigned int i = 0; i < va.size (); i++)
            tmp += (va)[i] * (va)[i];
    }
    void operator     () (void*,Shared_r < T1 > _va, Shared_r < T1 > _vb,
            Shared_rw < double >res)
    {
        helper::ReadAccessor<T1> vb = _vb.read ();
        helper::ReadAccessor<T1> va = _va.read ();

        double &tmp = res.access ();


        for (unsigned int i = 0; i < va.size (); i++)
            tmp += (va)[i] * (vb)[i];

    }
};







template < class DataTypes, class T1, class T2 > class vOpMinusEqualMult
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb,
            Shared_r < double >_f)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();
        const double &f = _f.read ();
        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            (vv2)[i] += (vb2)[i] * (-f);


        }
    }
};
template < class DataTypes, class T1, class T2 > class vOpSumMult
{
public:

    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb,
            Shared_r < double >_f, typename DataTypes::Real)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();
        const typename DataTypes::Real & f = _f.read ();
        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {

            vv2[i] *= f;
            (vv2)[i] += vb2[i];
        }
    }
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb,
            typename DataTypes::Real f)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();
        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {

            vv2[i] *= f;
            (vv2)[i] += vb2[i];
        }
    }
};
template < class DataTypes, class T1, class T2 > class vOpSum
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > vb,
            typename DataTypes::Real f)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();
        vv2.resize (vb2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            vv2[i] *= f;
            (vv2)[i] += vb2[i];
        }
    }
};

template <  class T1, class T2, class T3 > class vAdd
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T2 > va,
            Shared_r < T3 > vb)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> va2 = va.read ();
        helper::ReadAccessor<T3> vb2 = vb.read ();
        vv2.resize (va2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            vv2[i] = va2[i];
            (vv2)[i] += vb2[i];
        }
    }
};
template < class DataTypes, class T1, class T2 > class vOpVeqAplusBmultF
{
public:
    void operator     () (void*,Shared_rw < T1 > vv, Shared_r < T1 > va,
            Shared_r < T2 > vb, typename DataTypes::Real f)
    {
        helper::WriteAccessor<T1> vv2 = vv.access ();
        helper::ReadAccessor<T2> vb2 = vb.read ();
        helper::ReadAccessor<T1> va2 = va.read ();
        vv2.resize (va2.size ());
        for (unsigned int i = 0; i < vv2.size (); i++)
        {
            vv2[i] = va2[i];
            (vv2)[i] += vb2[i] * f;
        }
    }
};


}
};
#endif

