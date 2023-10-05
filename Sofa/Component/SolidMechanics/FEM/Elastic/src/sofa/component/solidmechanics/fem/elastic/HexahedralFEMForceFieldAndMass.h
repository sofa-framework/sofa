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


#include <sofa/component/solidmechanics/fem/elastic/HexahedralFEMForceField.h>
#include <sofa/core/behavior/Mass.h>

#include <sofa/core/topology/TopologyData.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/** Compute Finite Element forces based on hexahedral elements including continuum mass matrices
*/
template<class DataTypes>
class HexahedralFEMForceFieldAndMass : virtual public sofa::core::behavior::Mass<DataTypes>, virtual public HexahedralFEMForceField<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HexahedralFEMForceFieldAndMass,DataTypes), SOFA_TEMPLATE(sofa::core::behavior::Mass,DataTypes), SOFA_TEMPLATE(HexahedralFEMForceField,DataTypes));

    typedef HexahedralFEMForceField<DataTypes> HexahedralFEMForceFieldT;
    typedef sofa::core::behavior::Mass<DataTypes> MassT;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef typename HexahedralFEMForceFieldT::Mat33 Mat33;
    typedef typename HexahedralFEMForceFieldT::Displacement Displacement;
    typedef typename HexahedralFEMForceFieldT::VecElement VecElement;
    typedef typename HexahedralFEMForceFieldT::ElementStiffness ElementMass;
    typedef core::topology::BaseMeshTopology::Index Index;
    typedef typename HexahedralFEMForceFieldT::HexahedronInformation HexahedronInformation;
    typedef typename HexahedralFEMForceFieldT::ElementStiffness ElementStiffness;
    typedef typename HexahedralFEMForceFieldT::Element Element;


protected:
    HexahedralFEMForceFieldAndMass();
public:
    void init( ) override;
    void reinit( ) override;

    // -- Mass interface
     void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

    ///// WARNING this method only add diagonal elements in the given matrix !
    void addMToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    bool isDiagonal() const override { return _useLumpedMass.getValue(); }

    using HexahedralFEMForceFieldT::addKToMatrix;
    using MassT::addKToMatrix;
    ///// WARNING this method only add diagonal elements in the given matrix !
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    ///// WARNING this method only add diagonal elements in the given matrix !
    void addMBKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

     void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;

     void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    SReal getKineticEnergy(const core::MechanicalParams* /* mparams */, const DataVecDeriv& /*v*/)  const override ///< vMv/2 using dof->getV() override
    {
        msg_error() << "HexahedralFEMForceFieldAndMass<DataTypes>::getKineticEnergy not yet implemented"; 
        return 0;
    }

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    void draw(const core::visual::VisualParams* vparams) override;

    SReal getElementMass(sofa::Index index) const override;

    void setDensity(Real d) {_density.setValue( d );}
    Real getDensity() {return _density.getValue();}




protected:
    virtual void computeElementMasses( ); ///< compute the mass matrices
    Real integrateVolume( int signx, int signy, int signz, Real l0, Real l1, Real l2 );
    virtual void computeElementMass( ElementMass &Mass, Real& totalMass, const type::fixed_array<Coord,8> &nodes); ///< compute the mass matrix of an element

    void computeParticleMasses();

    void computeLumpedMasses();

protected:
    Data<Real> _density; ///< density == volumetric mass in english (kg.m-3)
    Data<bool> _useLumpedMass; ///< Does it use lumped masses?

    core::topology::HexahedronData<sofa::type::vector<ElementMass> > _elementMasses; ///< mass matrices per element
    core::topology::HexahedronData<sofa::type::vector<Real> > _elementTotalMass; ///< total mass per element

    core::topology::PointData<sofa::type::vector<Real> > _particleMasses; ///< masses per particle in order to compute gravity
    core::topology::PointData<sofa::type::vector<Coord> > _lumpedMasses; ///< masses per particle computed by lumping mass matrices
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API HexahedralFEMForceFieldAndMass<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::solidmechanics::fem::elastic
