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
#ifndef FRAME_FRAMEDIAGONALMASS_H
#define FRAME_FRAMEDIAGONALMASS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/Mass.h>
#include "FrameMass.h"
#include "Blending.h"
#include "initFrame.h"
#include <SofaBaseTopology/PointSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace mass
{
using sofa::defaulttype::FrameData;
using defaulttype::Quat;
using sofa::component::topology::PointSetTopologyContainer;

/** One different mass for each moving frame */
template <class DataTypes, class TMassType>
class FrameDiagonalMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameDiagonalMass,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef TMassType MassType;
    enum { N=DataTypes::spatial_dimensions };
    enum { InDerivDim=DataTypes::Deriv::total_size };
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef defaulttype::Mat<InDerivDim,InDerivDim,Real> MatInxIn;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef defaulttype::Vec<InDerivDim,Real> VecIn;
    typedef FrameData<DataTypes,true> FData;
    typedef typename FData::VecMass VecMass;
    typedef typename FData::MassVector MassVector;

    VecMass f_mass0;
    VecMass f_mass;

    /// to display the center of gravity of the system
    Data< float > showAxisSize;
    core::objectmodel::DataFileName fileMass;
    Data< float > damping;

    FrameDiagonalMass();
    ~FrameDiagonalMass();

    bool load(const char *filename);

    void clear();

    virtual void init();
    virtual void reinit();
    virtual void bwdInit();

    void addMass(const MassType& mass);

    void resize(int vsize);

    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    double getKineticEnergy(const VecDeriv& v) const;  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const VecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, core::MultiVecDerivId vid);

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal()
    {
        return true;
    };

    void draw(const core::visual::VisualParams* vparams);

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

protected:
    class Loader;

private:
    FData* frameData; // Storage and computation of the mass blocks

    void updateMass();
    void rotateMass();
    void computeRelRot (Mat33& relRot, const Quat& q, const Quat& q0);
    void rotateM( MatInxIn& M, const MatInxIn& M0, const Mat33& R);
    void QtoR( Mat33& M, const sofa::helper::Quater<Real>& q);
};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FRAME_FRAMEDIAGONALMASS_CPP)
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
