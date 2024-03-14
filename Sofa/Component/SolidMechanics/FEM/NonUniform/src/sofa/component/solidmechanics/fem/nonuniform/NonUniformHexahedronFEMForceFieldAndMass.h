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

#include <sofa/component/solidmechanics/fem/nonuniform/config.h>

#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceFieldAndMass.h>

namespace sofa::component::solidmechanics::fem::nonuniform
{

/** Need a SparseGridTopology with _sparseGrid->_nbVirtualFinerLevels >= this->_nbVirtualFinerLevels

@InProceedings{NPF06,
author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
title        = "Animating Shapes at Arbitrary Resolution with Non-Uniform Stiffness",
booktitle    = "Eurographics Workshop in Virtual Reality Interaction and Physical Simulation (VRIPHYS)",
month        = "nov",
year         = "2006",
organization = "Eurographics",
address      = "Madrid",
url          = "http://www-evasion.imag.fr/Publications/2006/NPF06"
}


*/

template<class DataTypes>
class NonUniformHexahedronFEMForceFieldAndMass : virtual public component::solidmechanics::fem::elastic::HexahedronFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NonUniformHexahedronFEMForceFieldAndMass,DataTypes), SOFA_TEMPLATE(component::solidmechanics::fem::elastic::HexahedronFEMForceFieldAndMass,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra VecElement;

    typedef component::solidmechanics::fem::elastic::HexahedronFEMForceFieldAndMass<DataTypes> HexahedronFEMForceFieldAndMassT;
    typedef component::solidmechanics::fem::elastic::HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;

    typedef typename HexahedronFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename HexahedronFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename HexahedronFEMForceFieldAndMassT::MassT MassT;
    typedef typename HexahedronFEMForceFieldAndMassT::ElementMass ElementMass;

    using Index = sofa::Index;

public:


    Data<int> d_nbVirtualFinerLevels; ///< use virtual finer levels, in order to compte non-uniform stiffness, only valid if the topology is a SparseGridTopology with enough VirtualFinerLevels.
    Data<bool> d_useMass; ///< Do we want to use this ForceField like a Mass? (or do we prefer using a separate Mass)
    Data<Real> d_totalMass;

    /// Link to be set to the topology container in the component graph. 
    SingleLink<NonUniformHexahedronFEMForceFieldAndMass<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
protected:
    NonUniformHexahedronFEMForceFieldAndMass();

public:

    void init() override;
    void reinit()  override { msg_warning() << "Non-uniform mechanical properties can't be updated, changes on mechanical properties (young, poisson, density) are not taken into account."; }

    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;
    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;


protected:


    static const float FINE_TO_COARSE[8][8][8]; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)
    /// add a matrix of a fine element to its englobing coarser matrix
    void addFineToCoarse( ElementStiffness& coarse, const ElementStiffness& fine, Index indice );

    /// condensate matrice from the (virtual) finest level to the actual mechanical level
    /// recursive function
    /// if level is the finest level, matrices are built as usual
    /// else  finer matrices are built by condensation and added to the current matrices by addFineToCoarse
    virtual void computeMechanicalMatricesByCondensation( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level);
    virtual void computeMechanicalMatricesByCondensation(); // call previous method for all elements


    void computeClassicalMechanicalMatrices( ElementStiffness &K, ElementMass &M, const Index elementIndice, int level);


    /// compute the hookean material matrix
    void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);

};


#if !defined(SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDANDMASS_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_NONUNIFORM_API NonUniformHexahedronFEMForceFieldAndMass<sofa::defaulttype::Vec3Types>;


#endif


} // namespace sofa::component::solidmechanics::fem::nonuniform
