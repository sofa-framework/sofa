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
#include <sofa/component/solidmechanics/fem/elastic/config.h>


#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.h>
#include <sofa/core/behavior/Mass.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/** Compute Finite Element forces based on hexahedral elements including continuum mass matrices
*/
template<class DataTypes>
class HexahedronFEMForceFieldAndMass : virtual public core::behavior::Mass<DataTypes>, virtual public HexahedronFEMForceField<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HexahedronFEMForceFieldAndMass,DataTypes), SOFA_TEMPLATE(sofa::core::behavior::Mass,DataTypes), SOFA_TEMPLATE(HexahedronFEMForceField,DataTypes));

    typedef HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;
    typedef sofa::core::behavior::Mass<DataTypes> MassT;

    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef VecCoord Vector;

    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;

    typedef typename HexahedronFEMForceFieldT::Mat33 Mat33;
    typedef typename HexahedronFEMForceFieldT::Displacement Displacement;
    typedef typename HexahedronFEMForceFieldT::VecElement VecElement;
    typedef typename HexahedronFEMForceFieldT::VecElementStiffness VecElementMass;
    typedef typename HexahedronFEMForceFieldT::ElementStiffness ElementMass;
    typedef type::vector<Real> MassVector;
    using Index = sofa::Index;

protected:
    HexahedronFEMForceFieldAndMass();
public:

    void init( ) override;
    void reinit( ) override;

    virtual void computeElementMasses( ); ///< compute the mass matrices
    virtual void computeElementMass( ElementMass &Mass, const type::fixed_array<Coord,8> &nodes, const Index elementIndice, SReal stiffnessFactor=1.0); ///< compute the mass matrix of an element
    Real integrateMass( int signx, int signy, int signz, Real l0, Real l1, Real l2 );

    // -- Mass interface
     void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

    void addMToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    bool isDiagonal() const override { return d_lumpedMass.getValue(); }

    using HexahedronFEMForceFieldT::addKToMatrix;
    using core::behavior::Mass<DataTypes>::addKToMatrix;
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override
    {
        HexahedronFEMForceFieldT::addKToMatrix(mparams, matrix);
    }

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix*) override;
    void buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices) override;

     void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;

     void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    SReal getKineticEnergy(const core::MechanicalParams*, const DataVecDeriv& /*v*/ ) const override ///< vMv/2 using dof->getV() override
    {
        msg_warning() << "HexahedronFEMForceFieldAndMass::getKineticEnergy() not implemented" << msgendl;
        return 0.0;
    }

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    SReal getElementMass(Index index) const override;
    // visual model

    void draw(const core::visual::VisualParams* vparams) override;

    virtual void initTextures() { }

    virtual void update() { }



    void setDensity(Real d) {d_density.setValue( d );}
    Real getDensity() {return d_density.getValue();}



protected :

    Data<VecElementMass> d_elementMasses; ///< mass matrices per element
    Data<Real> d_density; ///< density == volumetric mass in english (kg.m-3)
    Data<bool> d_lumpedMass; ///< Does it use lumped masses?

    MassVector _particleMasses; ///< masses per particle in order to compute gravity
    type::vector<Coord> _lumpedMasses; ///< masses per particle computed by lumping mass matrices


};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELDANDMASS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API HexahedronFEMForceFieldAndMass< defaulttype::Vec3Types >;

#endif

} // namespace sofa::component::solidmechanics::fem::elastic
