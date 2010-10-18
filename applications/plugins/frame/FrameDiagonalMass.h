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
#ifndef FRAME_FRAMEDIAGONALMASS_H
#define FRAME_FRAMEDIAGONALMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "AffineTypes.h"
#include "QuadraticTypes.h"
#include "FrameMass.h"
#include <sofa/component/mapping/FrameStorage.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include "initFrame.h"

namespace sofa
{

namespace component
{

namespace mass
{

using sofa::component::mapping::FrameStorage;
using namespace sofa::defaulttype;

template <class DataTypes, class TMassType>
class FrameDiagonalMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameDiagonalMass,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef TMassType MassType;
    enum { N=DataTypes::spatial_dimensions };
    enum { InDerivDim=DataTypes::deriv_total_size };
    typedef defaulttype::Mat<N,N,Real> Mat;
    //typedef defaulttype::Mat<3,1,Real> Mat31;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef defaulttype::Mat<3,6,Real> Mat36;
    typedef vector<Mat36> VMat36;
    typedef vector<VMat36> VVMat36;
    typedef defaulttype::Mat<3,8,Real> Mat38;
    typedef defaulttype::Mat<3,9,Real> Mat39;
    typedef defaulttype::Mat<3,InDerivDim,Real> Mat3xIn;
    typedef vector<Mat3xIn> VMat3xIn;
    typedef vector<VMat3xIn> VVMat3xIn;
    typedef defaulttype::Mat<4,4,Real> Mat44;
    //typedef defaulttype::Mat<6,1,Real> Mat61;
    typedef defaulttype::Mat<6,3,Real> Mat63;
    typedef defaulttype::Mat<InDerivDim,3,Real> MatInx3;
    typedef vector<MatInx3> VMatInx3;
    typedef vector<VMatInx3> VVMatInx3;
    typedef defaulttype::Mat<6,6,Real> Mat66;
    typedef defaulttype::Mat<InDerivDim,InDerivDim,Real> MatInxIn;
    typedef vector<Mat66> VMat66;
    typedef vector<VMat66> VVMat66;
    //typedef defaulttype::Mat<8,1,Real> Mat81;
    typedef defaulttype::Mat<8,3,Real> Mat83;
    typedef defaulttype::Mat<8,6,Real> Mat86;
    typedef vector<Mat86> VMat86;
    typedef defaulttype::Mat<8,8,Real> Mat88;
    typedef vector<Mat88> VMat88;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef vector<Vec3> VVec3;
    typedef vector<VVec3> VVVec3;
    typedef defaulttype::Vec<4,Real> Vec4;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef vector<Vec6> VVec6;
    typedef vector<VVec6> VVVec6;
    typedef defaulttype::Vec<8,Real> Vec8;
    typedef vector<double> VD;
    typedef FrameStorage<DataTypes, Real> FStorage;

    // In case of non 3D template
    typedef Vec<3,MassType>                            Vec3Mass;
    typedef StdVectorTypes< Vec3Mass, Vec3Mass, MassType > GeometricalTypes ; /// assumes the geometry object type is 3D

    typedef sofa::component::topology::PointData<MassType> VecMass;
    typedef helper::vector<MassType> MassVector;

    VecMass f_mass;
    VecMass f_mass0;
    /// the mass density used to compute the mass from a mesh topology and geometry
    Data< Real > m_massDensity; // Used to fix mass density of all the samples.

    /// to display the center of gravity of the system
    Data< bool > showCenterOfGravity;
    Data< float > showAxisSize;
    core::objectmodel::DataFileName fileMass;
    Data< float > damping;
    Data<bool> rotateMass;

protected:
    //VecMass masses;

    class Loader;

public:

    FrameDiagonalMass();

    ~FrameDiagonalMass();

    //virtual const char* getTypeName() const { return "FrameDiagonalMass"; }

    bool load(const char *filename);

    void clear();

    virtual void init();
    virtual void reinit();
    virtual void bwdInit();

    Real getMassDensity() const
    {
        return m_massDensity.getValue();
    }

    void setMassDensity(Real m)
    {
        m_massDensity.setValue(m);
    }


    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    double getKineticEnergy(const VecDeriv& v) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const VecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(double dt/*, defaulttype::BaseVector& v*/);

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal() {return true;};

    void draw();

    bool addBBox(double* minBBox, double* maxBBox);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const sofa::core::behavior::ForceField<DataTypes>* = NULL)
    {
        std::string name;
        name.append(DataTypes::Name());
        name.append(MassType::Name());
        return name;
    }

private:
    FStorage* dqStorage;
    VD* vol;
    VD* massDensity;
    VVMat3xIn* J;
    VVMat3xIn* J0;

    void updateMass ( MassType& mass, const VMat3xIn& J, const VD& vol, const VD& volmass );
    void computeRelRot ( Mat33& relRot, const Coord& xi, const Coord& xi0);
    void rotateM( MatInxIn& M, const MatInxIn& M0, const Mat33& R);
    void QtoR( Mat33& M, const sofa::helper::Quater<Real>& q);
};

#if defined(WIN32) && !defined(FRAME_FRAMEDIAGONALMASS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_FRAME_API FrameDiagonalMass<Rigid3dTypes,Frame3dMass>;
extern template class SOFA_FRAME_API FrameDiagonalMass<Affine3dTypes,Frame3x12dMass>;
extern template class SOFA_FRAME_API FrameDiagonalMass<Quadratic3dTypes,Frame3x30dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_FRAME_API FrameDiagonalMass<Rigid3fTypes,Frame3fMass>;
extern template class SOFA_FRAME_API FrameDiagonalMass<Affine3fTypes,Frame3x12fMass>;
extern template class SOFA_FRAME_API FrameDiagonalMass<Quadratic3fTypes,Frame3x30fMass>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
