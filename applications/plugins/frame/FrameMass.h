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
#ifndef FRAME_FRAMEMASS_H
#define FRAME_FRAMEMASS_H

#include <sofa/defaulttype/RigidTypes.h>
#include "AffineTypes.h"
#include "QuadraticTypes.h"
#include <sofa/simulation/common/Visitor.h>

namespace sofa
{

namespace defaulttype
{

template<int Nl, int Nc, typename real>
class FrameMass;

using sofa::simulation::Visitor;

//=============================================================================
// 3D Frames
//=============================================================================

template<int Nc, typename real>
class FrameMass<3, Nc, real>
{
public:
    //typedef real value_type;
    typedef real Real;
    typedef typename StdAffineTypes<3,Real>::Deriv AffineDeriv;
    typedef typename StdRigidTypes<3,Real>::Deriv RigidDeriv;
    typedef typename StdQuadraticTypes<3,Real>::Deriv QuadraticDeriv;
    typedef Mat<Nc,Nc,Real> MatMass;
    typedef Vec<Nc,Real> VecDOFs;
    typedef vector<double> VD;
    enum { InDerivDim=Nc};//StdRigidTypes<3,Real>::deriv_total_size };
    Real mass;
    MatMass inertiaMatrix;	      // Inertia matrix of the object
    MatMass inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    MatMass invInertiaMatrix;	  // inverse of inertiaMatrix
    MatMass invInertiaMassMatrix; // inverse of inertiaMassMatrix

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
    RigidDeriv operator * ( const RigidDeriv& a ) const
    {
        VecDOFs va, vma;
        va[0] = a.getVOrientation() [0];
        va[1] = a.getVOrientation() [1];
        va[2] = a.getVOrientation() [2];
        va[3] = a.getVCenter() [0];
        va[4] = a.getVCenter() [1];
        va[5] = a.getVCenter() [2];

        vma = inertiaMassMatrix * va;
        //std::cerr << "inertiaMassMatrix: " << inertiaMassMatrix << std::endl;

        RigidDeriv ma;
        ma.getVOrientation() [0] = vma[0];
        ma.getVOrientation() [1] = vma[1];
        ma.getVOrientation() [2] = vma[2];
        ma.getVCenter() [0] = vma[3];
        ma.getVCenter() [1] = vma[4];
        ma.getVCenter() [2] = vma[5];

        return ma;
    }

    /// compute a = f/m
    RigidDeriv operator / ( const RigidDeriv& f ) const
    {
        VecDOFs va, vma;
        vma[0] = f.getVOrientation() [0];
        vma[1] = f.getVOrientation() [1];
        vma[2] = f.getVOrientation() [2];
        vma[3] = f.getVCenter() [0];
        vma[4] = f.getVCenter() [1];
        vma[5] = f.getVCenter() [2];

        va = invInertiaMassMatrix * vma;
        //std::cerr << "invInertiaMassMatrix: " << invInertiaMassMatrix << std::endl;
        RigidDeriv a;
        a.getVOrientation() [0] = va[0];
        a.getVOrientation() [1] = va[1];
        a.getVOrientation() [2] = va[2];
        a.getVCenter() [0] = va[3];
        a.getVCenter() [1] = va[4];
        a.getVCenter() [2] = va[5];

        return a;
    }

    /// compute ma = M*a
    AffineDeriv operator * ( const AffineDeriv& a ) const
    {
        VecDOFs va, vma;
        va[0] = a.getVAffine() [0][0];
        va[1] = a.getVAffine() [0][1];
        va[2] = a.getVAffine() [0][2];
        va[3] = a.getVAffine() [1][0];
        va[4] = a.getVAffine() [1][1];
        va[5] = a.getVAffine() [1][2];
        va[6] = a.getVAffine() [2][0];
        va[7] = a.getVAffine() [2][1];
        va[8] = a.getVAffine() [2][2];
        va[9] = a.getVCenter() [0];
        va[10] = a.getVCenter() [1];
        va[11] = a.getVCenter() [2];

        vma = inertiaMassMatrix * va;
        //std::cerr << "inertiaMassMatrix: " << inertiaMassMatrix << std::endl;

        AffineDeriv ma;
        ma.getVAffine() [0][0] = vma[0];
        ma.getVAffine() [0][1] = vma[1];
        ma.getVAffine() [0][2] = vma[2];
        ma.getVAffine() [1][0] = vma[3];
        ma.getVAffine() [1][1] = vma[4];
        ma.getVAffine() [1][2] = vma[5];
        ma.getVAffine() [2][0] = vma[6];
        ma.getVAffine() [2][1] = vma[7];
        ma.getVAffine() [2][2] = vma[8];
        ma.getVCenter() [0] = vma[9];
        ma.getVCenter() [1] = vma[10];
        ma.getVCenter() [2] = vma[11];

        return ma;
    }

    /// compute a = f/m
    AffineDeriv operator / ( const AffineDeriv& f ) const
    {
        VecDOFs va, vma;
        vma[0] = f.getVAffine() [0][0];
        vma[1] = f.getVAffine() [0][1];
        vma[2] = f.getVAffine() [0][2];
        vma[3] = f.getVAffine() [1][0];
        vma[4] = f.getVAffine() [1][1];
        vma[5] = f.getVAffine() [1][2];
        vma[6] = f.getVAffine() [2][0];
        vma[7] = f.getVAffine() [2][1];
        vma[8] = f.getVAffine() [2][2];
        vma[9] = f.getVCenter() [0];
        vma[10] = f.getVCenter() [1];
        vma[11] = f.getVCenter() [2];

        va = invInertiaMassMatrix * vma;
        //std::cerr << "invInertiaMassMatrix: " << invInertiaMassMatrix << std::endl;
        AffineDeriv a;
        a.getVAffine() [0][0] = va[0];
        a.getVAffine() [0][1] = va[1];
        a.getVAffine() [0][2] = va[2];
        a.getVAffine() [1][0] = va[3];
        a.getVAffine() [1][1] = va[4];
        a.getVAffine() [1][2] = va[5];
        a.getVAffine() [2][0] = va[6];
        a.getVAffine() [2][1] = va[7];
        a.getVAffine() [2][2] = va[8];
        a.getVCenter() [0] = va[9];
        a.getVCenter() [1] = va[10];
        a.getVCenter() [2] = va[11];

        return a;
    }

    /// compute ma = M*a
    QuadraticDeriv operator * ( const QuadraticDeriv& a ) const
    {
        const unsigned int& dim = (Nc-3)/3;
        VecDOFs va, vma;
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                va[dim*i+j] = a.getVQuadratic() [i][j];
        va[3*dim  ] = a.getVCenter() [0];
        va[3*dim+1] = a.getVCenter() [1];
        va[3*dim+2] = a.getVCenter() [2];

        vma = inertiaMassMatrix * va;
        //std::cerr << "inertiaMassMatrix: " << inertiaMassMatrix << std::endl;

        QuadraticDeriv ma;
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                ma.getVQuadratic()[i][j] = vma[dim*i+j];
        ma.getVCenter()[0] = vma[3*dim];
        ma.getVCenter()[1] = vma[3*dim+1];
        ma.getVCenter()[2] = vma[3*dim+2];

        return ma;
    }

    /// compute a = f/m
    QuadraticDeriv operator / ( const QuadraticDeriv& f ) const
    {
        const unsigned int& dim = (Nc-3)/3;
        VecDOFs va, vma;
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                vma[dim*i+j] = f.getVQuadratic() [i][j];
        vma[3*dim  ] = f.getVCenter() [0];
        vma[3*dim+1] = f.getVCenter() [1];
        vma[3*dim+2] = f.getVCenter() [2];

        va = invInertiaMassMatrix * vma;
        //std::cerr << "invInertiaMassMatrix: " << invInertiaMassMatrix << std::endl;

        QuadraticDeriv a;
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                a.getVQuadratic()[i][j] = va[dim*i+j];
        a.getVCenter()[0] = va[3*dim  ];
        a.getVCenter()[1] = va[3*dim+1];
        a.getVCenter()[2] = va[3*dim+2];

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

    inline friend std::ostream& operator << ( std::ostream& out, const FrameMass<3, Nc, real>& m )
    {
        out<<m.mass;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> ( std::istream& in, FrameMass<3, Nc, real>& m )
    {
        in>>m.mass;
        in>>m.inertiaMatrix;
        return in;
    }

    static const char* Name();

};


template<int Nl, int Nc, typename real>
inline typename StdRigidTypes<Nl, real>::Deriv operator* ( const typename StdRigidTypes<Nl,real>::Deriv& d, const FrameMass<Nl, Nc, real>& m )
{
    return m * d;
}

template<int Nl, int Nc, typename real>
inline typename StdRigidTypes<Nl, real>::Deriv operator/ ( const typename StdRigidTypes<Nl, real>::Deriv& d, const FrameMass<Nl, Nc, real>& m )
{
    return m / d;
}

template<int Nl, int Nc, typename real>
inline typename StdAffineTypes<Nl, real>::Deriv operator* ( const typename StdAffineTypes<Nl,real>::Deriv& d, const FrameMass<Nl, Nc, real>& m )
{
    return m * d;
}

template<int Nl, int Nc, typename real>
inline typename StdAffineTypes<Nl, real>::Deriv operator/ ( const typename StdAffineTypes<Nl, real>::Deriv& d, const FrameMass<Nl, Nc, real>& m )
{
    return m / d;
}

template<int Nl, int Nc, typename real>
inline typename StdQuadraticTypes<Nl, real>::Deriv operator* ( const typename StdQuadraticTypes<Nl,real>::Deriv& d, const FrameMass<Nl, Nc, real>& m )
{
    return m * d;
}

template<int Nl, int Nc, typename real>
inline typename StdQuadraticTypes<Nl, real>::Deriv operator/ ( const typename StdQuadraticTypes<Nl, real>::Deriv& d, const FrameMass<Nl, Nc, real>& m )
{
    return m / d;
}

typedef FrameMass<3,6,double> Frame3x6dMass;
typedef FrameMass<3,6,float> Frame3x6fMass;
typedef FrameMass<3,12,double> Frame3x12dMass;
typedef FrameMass<3,12,float> Frame3x12fMass;
typedef FrameMass<3,30,double> Frame3x30dMass;
typedef FrameMass<3,30,float> Frame3x30fMass;

template<> inline const char* defaulttype::Frame3x6dMass::Name() { return "Frame3x6dMass"; }
template<> inline const char* defaulttype::Frame3x6fMass::Name() { return "Frame3x6fMass"; }
template<> inline const char* defaulttype::Frame3x12dMass::Name() { return "Frame3x12dMass"; }
template<> inline const char* defaulttype::Frame3x12fMass::Name() { return "Frame3x12fMass"; }
template<> inline const char* defaulttype::Frame3x30dMass::Name() { return "Frame3x30dMass"; }
template<> inline const char* defaulttype::Frame3x30fMass::Name() { return "Frame3x30fMass"; }

#ifdef SOFA_FLOAT
typedef Frame3x6fMass Frame3x6Mass;
typedef Frame3x12fMass Frame3x12Mass;
typedef Frame3x30fMass Frame3x130Mass;
#else
typedef Frame3x6dMass Frame3x6Mass;
typedef Frame3x12dMass Frame3x12Mass;
typedef Frame3x30dMass Frame3x30Mass;
#endif

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Frame3x6fMass >
{
    static const char* name()
    {
        return "Frame3x6fMass";
    }
};
template<> struct DataTypeName< defaulttype::Frame3x6dMass >
{
    static const char* name()
    {
        return "Frame3x6dMass";
    }
};

template<> struct DataTypeName< defaulttype::Frame3x12fMass >
{
    static const char* name()
    {
        return "Frame3x12fMass";
    }
};
template<> struct DataTypeName< defaulttype::Frame3x12dMass >
{
    static const char* name()
    {
        return "Frame3x12dMass";
    }
};

template<> struct DataTypeName< defaulttype::Frame3x30fMass >
{
    static const char* name()
    {
        return "Frame3x30fMass";
    }
};
template<> struct DataTypeName< defaulttype::Frame3x30dMass >
{
    static const char* name()
    {
        return "Frame3x30dMass";
    }
};

/// \endcond


} // namespace defaulttype

} // namespace sofa

#endif

