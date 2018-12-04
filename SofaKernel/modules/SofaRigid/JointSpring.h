/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_JOINTSPRING_H
#define SOFA_JOINTSPRING_H
#include "config.h"

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/// see Template friend operators
/// in https://en.cppreference.com/w/cpp/language/friend
template<typename DataTypes>
class JointSpring; // forward declare to make function declaration possible

template<typename DataTypes>
std::istream& operator>> ( std::istream& in, JointSpring<DataTypes>& s );
template<typename DataTypes>
std::ostream& operator<< ( std::ostream& out, const JointSpring<DataTypes>& s );
//////////////////////////////////////////////////////////

/// JOINTSPRING
template<typename DataTypes>
class JointSpring
{
public:

    typedef typename DataTypes::Coord    Coord   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Vec<N,Real> Vector;

    /// Constructors
    JointSpring(int m1 = 0,int m2 = 0,
                Real softKst = 0, Real hardKst = 10000, Real softKsr = 0, Real hardKsr = 10000, Real blocKsr = 100,
                Real axmin = -100000, Real axmax = 100000, Real aymin = -100000, Real aymax = 100000, Real azmin = -100000, Real azmax= 100000,
                Real kd = 0);

    /// Attributes
    int  m1, m2;                    /// the two extremities of the spring: masses m1 and m2
    Real kd;                        /// damping factor
    Vector torsion;                 /// torsion of the springs in axis/angle format
    Vector lawfulTorsion;           /// projected torsion in allowed angles
    Vector KT;                      /// linear stiffness
    Vector KR;                      /// angular stiffness
    defaulttype::Quat ref;          /// referential of the spring (p1) to use it in addSpringDForce()

    Vector  initTrans;              /// offset length of the spring
    defaulttype::Quat initRot;      /// offset orientation of the spring

    sofa::defaulttype::Vec<6,bool> freeMovements;	///defines the axis where the movements is free. (0,1,2)--> translation axis (3,4,5)-->rotation axis
    Real softStiffnessTrans;                        ///stiffness to apply on axis where the translations are free (default 0.0)
    Real hardStiffnessTrans;                        ///stiffness to apply on axis where the translations are forbidden (default 10000.0)
    Real softStiffnessRot;                          ///stiffness to apply on axis where the rotations are free (default 0.0)
    Real hardStiffnessRot;                          ///stiffness to apply on axis where the rotations are forbidden (default 10000.0)
    Real blocStiffnessRot;                          ///stiffness to apply on axis where the rotations are bloqued (=hardStiffnessRot/100)
    bool needToInitializeTrans;
    bool needToInitializeRot;

    sofa::defaulttype::Vec<6,Real> limitAngles; ///limit angles on rotation axis (default no limit)


    /// Accessors
    Real getHardStiffnessRotation() {return hardStiffnessRot;}
    Real getSoftStiffnessRotation() {return softStiffnessRot;}
    Real getHardStiffnessTranslation() {return hardStiffnessTrans;}
    Real getSoftStiffnessTranslation() {return softStiffnessTrans;}
    Real getBlocStiffnessRotation() { return blocStiffnessRot; }
    sofa::defaulttype::Vec<6,Real> getLimitAngles() { return limitAngles;}
    sofa::defaulttype::Vec<6,bool> getFreeAxis() { return freeMovements;}
    Vector getInitLength() { return initTrans; }
    defaulttype::Quat getInitOrientation() { return initRot; }

    /// Affectors
    void setHardStiffnessRotation(Real ksr) {	  hardStiffnessRot = ksr;  }
    void setSoftStiffnessRotation(Real ksr) {	  softStiffnessRot = ksr;  }
    void setHardStiffnessTranslation(Real kst) { hardStiffnessTrans = kst;  }
    void setSoftStiffnessTranslation(Real kst) { softStiffnessTrans = kst;  }
    void setBlocStiffnessRotation(Real ksb) {	  blocStiffnessRot = ksb;  }
    void setLimitAngles(const sofa::defaulttype::Vec<6,Real>& lims)
    {
        limitAngles = lims;
        if(lims[0]==lims[1]) freeMovements[3]=false;
        if(lims[2]==lims[3]) freeMovements[4]=false;
        if(lims[4]==lims[5]) freeMovements[5]=false;
    }
    void setLimitAngles(Real minx, Real maxx, Real miny, Real maxy, Real minz, Real maxz)
    {
        limitAngles = sofa::defaulttype::Vec<6,Real>(minx, maxx, miny, maxy, minz, maxz);
        if(minx==maxx) freeMovements[3]=false;
        if(miny==maxy) freeMovements[4]=false;
        if(minz==maxz) freeMovements[5]=false;
    }
    void setInitLength( const Vector& l) { initTrans=l; }
    void setInitOrientation( const defaulttype::Quat& o) { initRot=o; }
    void setInitOrientation( const Vector& o) { initRot=defaulttype::Quat::createFromRotationVector(o); }
    void setFreeAxis(const sofa::defaulttype::Vec<6,bool>& axis) { freeMovements = axis; }
    void setFreeAxis(bool isFreeTx, bool isFreeTy, bool isFreeTz, bool isFreeRx, bool isFreeRy, bool isFreeRz)
    {
        freeMovements = sofa::defaulttype::Vec<6,bool>(isFreeTx, isFreeTy, isFreeTz, isFreeRx, isFreeRy, isFreeRz);
    }
    void setDamping(Real _kd) {  kd = _kd;	  }

    /// see https://en.cppreference.com/w/cpp/language/friend
    friend std::istream& operator>> <>( std::istream& in, JointSpring& s );
    friend std::ostream& operator<< <>( std::ostream& out, const JointSpring& s );
    //////////////////////////////////////////////////////////

};

#if  !defined(SOFA_JOINTSPRING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RIGID_API JointSpring<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API JointSpring<defaulttype::Rigid3fTypes>;
#endif
#endif
} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif /* #define SOFA_JOINTSPRING_H */
