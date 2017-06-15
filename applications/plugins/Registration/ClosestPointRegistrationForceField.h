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
#ifndef CLOSESTPOINTREGISTRATIONFORCEFIELD_H
#define CLOSESTPOINTREGISTRATIONFORCEFIELD_H

#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/helper/kdTree.h>

#include <set>
#include <Registration/config.h>

#ifdef WIN32
#include <process.h>
#else
#include <pthread.h>
#endif

namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
class ClosestPointRegistrationForceFieldInternalData
{
public:
};


template<class DataTypes>
class SOFA_REGISTRATION_API ClosestPointRegistrationForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ClosestPointRegistrationForceField,DataTypes),SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

    typedef helper::fixed_array <unsigned int,3> tri;
    typedef helper::kdTree<Coord> KDT;
    typedef typename KDT::distanceSet distanceSet;
    typedef typename KDT::distanceToPoint distanceToPoint;

public:
    ClosestPointRegistrationForceField(core::behavior::MechanicalState<DataTypes> *mm = NULL);
    virtual ~ClosestPointRegistrationForceField();

    core::behavior::MechanicalState<DataTypes>* getObject() { return this->mstate; }

    // -- ForceField interface
    void reinit();
    void init();
    void addForce(const core::MechanicalParams* /*mparams*/,DataVecDeriv& f , const DataVecCoord& x , const DataVecDeriv& v);
    void addDForce(const core::MechanicalParams* mparams ,DataVecDeriv&   df , const DataVecDeriv&   dx);
    SReal getPotentialEnergy(const core::MechanicalParams* ,const DataVecCoord&) const { return m_potentialEnergy; }
    void addKToMatrix( const core::MechanicalParams* mparams,const sofa::core::behavior::MultiMatrixAccessor* matrix);

    Real getStiffness() const{ return ks.getValue(); }
    Real getDamping() const{ return kd.getValue(); }
    void setStiffness(Real _ks){ ks.setValue(_ks); }
    void setDamping(Real _kd){ kd.setValue(_kd); }
    Real getArrowSize() const{return showArrowSize.getValue();}
    void setArrowSize(float s){showArrowSize.setValue(s);}
    int getDrawMode() const{return drawMode.getValue();}
    void setDrawMode(int m){drawMode.setValue(m);}

    void draw(const core::visual::VisualParams* vparams);


protected :
    void updateClosestPoints();

    VecCoord closestPos;
    helper::vector<unsigned int>  cnt;
    SReal m_potentialEnergy;

    Real min,max;

    Data<Real> ks;
    Data<Real> kd;
    Data<unsigned int> cacheSize;
    Data<Real> blendingFactor;
    Data<Real> outlierThreshold;
    Data<Real> normalThreshold;
    Data<bool> projectToPlane;
    Data<bool> rejectBorders;
    Data<bool> rejectOutsideBbox;
    defaulttype::BoundingBox targetBbox;

    // source mesh data
    Data< helper::vector< tri > > sourceTriangles;
    Data< VecCoord > sourceNormals;
    helper::vector< distanceSet >  closestSource; // CacheSize-closest target points from source
    helper::vector< distanceToPoint > cacheThresh_max;	helper::vector< distanceToPoint > cacheThresh_min; VecCoord previousX; // storage for cache acceleration
    KDT sourceKdTree;
    helper::vector< bool > sourceBorder;
    helper::vector< bool > sourceIgnored;  // flag ignored vertices
    helper::vector< bool > targetIgnored;  // flag ignored vertices
    void initSource(); // built k-d tree and identify border vertices

    // target mesh data
    Data< VecCoord > targetPositions;
    Data< VecCoord > targetNormals;
    Data< helper::vector< tri > > targetTriangles;
    helper::vector< distanceSet >  closestTarget; // CacheSize-closest source points from target
    KDT targetKdTree;
    helper::vector< bool > targetBorder;
    void initTarget();  // built k-d tree and identify border vertices

    Data<float> showArrowSize;
    Data<int> drawMode; //Draw Mode: 0=Line - 1=Cylinder - 2=Arrow
    Data<bool> drawColorMap;
    Data<bool> theCloserTheStiffer;

    void detectBorder(helper::vector<bool> &border,const helper::vector< tri > &triangles);
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(CLOSESTPOINTREGISTRATIONFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_REGISTRATION_API ClosestPointRegistrationForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_REGISTRATION_API ClosestPointRegistrationForceField<defaulttype::Vec3fTypes>;
#endif
#endif


} //

} //

} // namespace sofa

#endif
