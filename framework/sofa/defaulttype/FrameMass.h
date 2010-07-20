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
#ifndef SOFA_COMPONENT_MASS_FRAMEMASS_H
#define SOFA_COMPONENT_MASS_FRAMEMASS_H

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Visitor.h>

namespace sofa
{

namespace defaulttype
{

template<int N, typename real>
class FrameMass;

using sofa::simulation::Visitor;

//=============================================================================
// 3D Frames
//=============================================================================

template<typename real>
class FrameMass<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef typename StdRigidTypes<3,Real>::VecCoord VecCoord;
    typedef typename StdRigidTypes<3,Real>::VecDeriv VecDeriv;
    typedef typename StdRigidTypes<3,Real>::Coord Coord;
    typedef typename StdRigidTypes<3,Real>::Deriv Deriv;
    typedef Mat<36,6,Real> Mat36;
    typedef Mat<6,6,Real> Mat66;
    typedef Vec<6,Real> Vec6;
    typedef vector<double> VD;
    Real mass;
    Mat66 inertiaMatrix;	      // Inertia matrix of the object
    Mat66 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat66 invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat66 invInertiaMassMatrix; // inverse of inertiaMassMatrix

    FrameMass ( Real m=1 )
    {
        mass = m;
        //recalc();
    }

    void operator= ( Real m )
    {
        mass = m;
        //recalc();
    }

    // operator to cast to const Real
    operator const Real() const
    {
        return mass;
    }

    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        invInertiaMatrix.invert ( inertiaMatrix );
        /*
        bool nullMatrix = true;
        for ( int i = 0; i < 6; i++ )
          for ( int j = 0; j < 6; j++ )
            if ( invInertiaMatrix[i][j] != 0 ) nullMatrix = false;
        if ( nullMatrix )
          for ( int i = 0; i < 6; i++ )
            invInertiaMatrix[i][i] = 1.0;
        	*/
        invInertiaMassMatrix.invert ( inertiaMassMatrix );
        /*
        nullMatrix = true;
        for ( int i = 0; i < 6; i++ )
          for ( int j = 0; j < 6; j++ )
            if ( invInertiaMassMatrix[i][j] != 0 ) nullMatrix = false;
        if ( nullMatrix )
          for ( int i = 0; i < 6; i++ )
            invInertiaMassMatrix[i][i] = mass;
        	*/
    }

    /// compute ma = M*a
    Deriv operator * ( const Deriv& a ) const
    {
        Vec6 va, vma;
        va[0] = a.getVOrientation() [0];
        va[1] = a.getVOrientation() [1];
        va[2] = a.getVOrientation() [2];
        va[3] = a.getVCenter() [0];
        va[4] = a.getVCenter() [1];
        va[5] = a.getVCenter() [2];

        vma = inertiaMassMatrix * va;
        //std::cerr << "inertiaMassMatrix: " << inertiaMassMatrix << std::endl;

        Deriv ma;
        ma.getVOrientation() [0] = vma[0];
        ma.getVOrientation() [1] = vma[1];
        ma.getVOrientation() [2] = vma[2];
        ma.getVCenter() [0] = vma[3];
        ma.getVCenter() [1] = vma[4];
        ma.getVCenter() [2] = vma[5];

        return ma;
    }

    /// compute a = f/m
    Deriv operator / ( const Deriv& f ) const
    {
        Vec6 va, vma;
        vma[0] = f.getVOrientation() [0];
        vma[1] = f.getVOrientation() [1];
        vma[2] = f.getVOrientation() [2];
        vma[3] = f.getVCenter() [0];
        vma[4] = f.getVCenter() [1];
        vma[5] = f.getVCenter() [2];

        va = invInertiaMassMatrix * vma;
        //std::cerr << "invInertiaMassMatrix: " << invInertiaMassMatrix << std::endl;
        Deriv a;
        a.getVOrientation() [0] = va[0];
        a.getVOrientation() [1] = va[1];
        a.getVOrientation() [2] = va[2];
        a.getVCenter() [0] = va[3];
        a.getVCenter() [1] = va[4];
        a.getVCenter() [2] = va[5];

        return a;
    }

    void operator *= ( Real fact )
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }
    void operator /= ( Real fact )
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const FrameMass<3, real>& m )
    {
        out<<m.mass;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> ( std::istream& in, FrameMass<3, real>& m )
    {
        in>>m.mass;
        in>>m.inertiaMatrix;
        return in;
    }

    static const char* Name();

};


template<int N, typename real>
inline typename StdRigidTypes<N, real>::Deriv operator* ( const typename StdRigidTypes<N, real>::Deriv& d, const FrameMass<N,real>& m )
{
    return m * d;
}

template<int N, typename real>
inline typename StdRigidTypes<N, real>::Deriv operator/ ( const typename StdRigidTypes<N, real>::Deriv& d, const FrameMass<N, real>& m )
{
    return m / d;
}

typedef FrameMass<3,double> Frame3dMass;
typedef FrameMass<3,float> Frame3fMass;

template<> inline const char* defaulttype::Frame3dMass::Name() { return "Frame3dMass"; }
template<> inline const char* defaulttype::Frame3fMass::Name() { return "Frame3fMass"; }

#ifdef SOFA_FLOAT
typedef Frame3fMass Frame3Mass;
#else
typedef Frame3dMass Frame3Mass;
#endif

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Frame3fMass >
{
    static const char* name()
    {
        return "Frame3fMass";
    }
};
template<> struct DataTypeName< defaulttype::Frame3dMass >
{
    static const char* name()
    {
        return "Frame3dMass";
    }
};

/// \endcond


} // namespace defaulttype

} // namespace sofa

#endif

