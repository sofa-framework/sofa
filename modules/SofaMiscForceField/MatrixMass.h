/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_MATRIXMASS_H
#define SOFA_COMPONENT_MASS_MATRIXMASS_H
#include "config.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>

namespace sofa
{



namespace component
{

namespace mass
{


/**
Mass Matrices. By default a diagonal matrix is created with diagonal to _defaultValue. Else matrices have to be given in the .scn file.
It is possible to use lumped matrices.
*/

template <class DataTypes, class MassType>
class [[deprecated("Class MatrixMass is deprecated and will be removed after 19.12")]] MatrixMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MatrixMass,DataTypes,MassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef helper::vector<MassType> VecMass;


    Data< VecMass > f_mass; ///< values of the particles masses
    Data< bool >    _lumped;
    Data< Real >    _defaultValue; ///< real default value

    VecMass _lumpedMasses; ///< lumped mass matrices

    const VecMass* _usedMassMatrices; ///< what VecMass is used to represent matrices ? f_mass.getValue() or _lumpedMasses ?

protected:
    MatrixMass();
    ~MatrixMass();

public:

    void clear();
    //void addMass(const MassType& mass);
    void resize(int vsize);


    void init() override;
    void reinit() override;


    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& v) const override;  ///< vMv/2 using dof->getV() override

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;   ///< Mgx potential in a uniform gravity field, null at origin

    defaulttype::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)+Iw) override

    /// Add Mass contribution to global Matrix assembling
    //void addMToMatrix(defaulttype::BaseMatrix * mat, SReal mFact, unsigned int &offset);
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    SReal getElementMass(unsigned int index) const override;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const override;

protected:
    MassType diagonalMass( const Real& m ); ///< return a diagonal matrix mass with value m on all the diagonal
    MassType lump( const MassType& m ); ///< lump the matrix m, ie sum line on diagonal
    void lumpMatrices( ); ///< lump all mass matrices
    void defaultDiagonalMatrices( ); ///< compute default diagonal matrices
    bool _usingDefaultDiagonalMatrices; ///< default diagonal matrices are used

};

#if  !defined(SOFA_COMPONENT_MASS_MATRIXMASS_CPP)
extern template class MatrixMass<defaulttype::Vec3Types, defaulttype::Mat3x3d>;
extern template class MatrixMass<defaulttype::Vec2Types, defaulttype::Mat2x2d>;
extern template class MatrixMass<defaulttype::Vec1Types, defaulttype::Mat1x1d>;

#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
