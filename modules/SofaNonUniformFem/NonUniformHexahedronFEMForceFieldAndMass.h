/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDANDMASS_H
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDANDMASS_H
#include "config.h"


#include <SofaGeneralSimpleFem/HexahedronFEMForceFieldAndMass.h>

namespace sofa
{

namespace component
{

namespace forcefield
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
class NonUniformHexahedronFEMForceFieldAndMass : virtual public HexahedronFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NonUniformHexahedronFEMForceFieldAndMass,DataTypes), SOFA_TEMPLATE(HexahedronFEMForceFieldAndMass,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

#ifdef SOFA_NEW_HEXA
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra VecElement;
#else
    typedef sofa::core::topology::BaseMeshTopology::SeqCubes VecElement;
#endif

    typedef HexahedronFEMForceFieldAndMass<DataTypes> HexahedronFEMForceFieldAndMassT;
    typedef HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;

    typedef typename HexahedronFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename HexahedronFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename HexahedronFEMForceFieldAndMassT::MassT MassT;
    typedef typename HexahedronFEMForceFieldAndMassT::ElementMass ElementMass;

public:


    Data<int> _nbVirtualFinerLevels; ///< use virtual finer levels, in order to compte non-uniform stiffness, only valid if the topology is a SparseGridTopology with enough VirtualFinerLevels.
    Data<bool> _useMass; ///< Do we want to use this ForceField like a Mass? (or do we prefer using a separate Mass)
    Data<Real> _totalMass;
protected:
    NonUniformHexahedronFEMForceFieldAndMass()
        : HexahedronFEMForceFieldAndMassT()
        , _nbVirtualFinerLevels(initData(&_nbVirtualFinerLevels,0,"nbVirtualFinerLevels","use virtual finer levels, in order to compte non-uniform stiffness"))
        , _useMass(initData(&_useMass,true,"useMass","Using this ForceField like a Mass? (rather than using a separated Mass)"))
        , _totalMass(initData(&_totalMass,(Real)0.0,"totalMass",""))
    {
    }

public:

    virtual void init() override;
    virtual void reinit()  override { serr<<"WARNING : non-uniform mechanical properties can't be updated, changes on mechanical properties (young, poisson, density) are not taken into account."<<sendl; }

    virtual void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;
    virtual void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;


protected:


    static const float FINE_TO_COARSE[8][8][8]; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)
    /// add a matrix of a fine element to its englobing coarser matrix
    void addFineToCoarse( ElementStiffness& coarse, const ElementStiffness& fine, int indice );

    /// condensate matrice from the (virtual) finest level to the actual mechanical level
    /// recursive function
    /// if level is the finest level, matrices are built as usual
    /// else  finer matrices are built by condensation and added to the current matrices by addFineToCoarse
    virtual void computeMechanicalMatricesByCondensation( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level);
    virtual void computeMechanicalMatricesByCondensation(); // call previous method for all elements


    void computeClassicalMechanicalMatrices( ElementStiffness &K, ElementMass &M, const int elementIndice, int level);


    /// compute the hookean material matrix
    void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDANDMASS_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_NON_UNIFORM_FEM_API NonUniformHexahedronFEMForceFieldAndMass<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_NON_UNIFORM_FEM_API NonUniformHexahedronFEMForceFieldAndMass<sofa::defaulttype::Vec3fTypes>;
#endif

#endif


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
