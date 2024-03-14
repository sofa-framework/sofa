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

#include <sofa/component/solidmechanics/fem/elastic/HexahedralFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedralFEMForceFieldAndMass.h>
#include <sofa/component/topology/container/dynamic/MultilevelHexahedronSetTopologyContainer.h>
#include <sofa/core/topology/TopologyChange.h>

namespace sofa::component::solidmechanics::fem::nonuniform
{

/**

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

indices ordering (same as in HexahedronSetTopology):

     Y  7---------6
     ^ /         /|
     |/    Z    / |
     3----^----2  |
     |   /     |  |
     |  4------|--5
     | /       | /
     |/        |/
     0---------1-->X

*/

template<class DataTypes>
class NonUniformHexahedralFEMForceFieldAndMass : virtual public component::solidmechanics::fem::elastic::HexahedralFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NonUniformHexahedralFEMForceFieldAndMass, DataTypes), SOFA_TEMPLATE(component::solidmechanics::fem::elastic::HexahedralFEMForceFieldAndMass, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef component::solidmechanics::fem::elastic::HexahedralFEMForceFieldAndMass<DataTypes> HexahedralFEMForceFieldAndMassT;
    typedef component::solidmechanics::fem::elastic::HexahedralFEMForceField<DataTypes> HexahedralFEMForceFieldT;

    typedef typename HexahedralFEMForceFieldAndMassT::VecElement VecElement;
    typedef typename HexahedralFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename HexahedralFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename HexahedralFEMForceFieldAndMassT::MassT MassT;
    typedef typename HexahedralFEMForceFieldAndMassT::ElementMass ElementMass;
    typedef typename HexahedralFEMForceFieldAndMassT::Element Element;

    typedef typename type::Mat<8, 8, Real> Mat88;
    typedef typename type::Vec<3, int> Vec3i;


protected:
    NonUniformHexahedralFEMForceFieldAndMass();
public:
    void init() override;
    void reinit() override;

    // handle topological changes
    void handleTopologyChange(core::topology::Topology*) override;

protected:
    /// condensate matrice from finest level to the actual mechanical level
    virtual void computeMechanicalMatricesByCondensation(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,const int elementIndex);

//  virtual void computeCorrection(ElementMass& /*M*/) {};

    void initLarge(const int i);

    void initPolar(const int i);

private:

    void handleHexaAdded(const core::topology::HexahedraAdded&);
    void handleHexaRemoved(const core::topology::HexahedraRemoved&);
    void handleMultilevelModif(const component::topology::container::dynamic::MultilevelModification&);


    void computeHtfineH(const Mat88& H, const ElementStiffness& fine, ElementStiffness& HtfineH ) const;
    void addHtfineHtoCoarse(const Mat88& H, const ElementStiffness& fine, ElementStiffness& coarse ) const;
    void subtractHtfineHfromCoarse(const Mat88& H, const ElementStiffness& fine, ElementStiffness& coarse ) const;

    void computeMechanicalMatricesByCondensation_IntervalAnalysis(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const unsigned int level,
        const std::set<Vec3i>& voxels) const;

    void computeMechanicalMatricesByCondensation_Recursive(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const unsigned int level,
        const unsigned int startIdx,
        const std::set<unsigned int>& fineChildren) const;

    void computeMechanicalMatricesByCondensation_Direct(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const unsigned int level,
        const std::set<Vec3i>& voxels) const;


    int ijk2octree(const int i, const int j, const int k) const;
    void octree2ijk(const int octreeIdx, int &i, int &j, int &k) const;
    Vec3i octree2voxel(const int octreeIdx) const;

    // [level][childId][childNodeId][parentNodeId] -> weight
    type::vector< type::vector< Mat88 > > _H; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)

    typedef struct
    {
        MaterialStiffness	C;	// Mat<6, 6, Real>
        ElementStiffness	K;	// Mat<24, 24, Real>
        ElementMass		M;	// Mat<24, 24, Real>
        Real			mass;
    } Material;

    Material _material; // TODO: enable combination of multiple materials

    component::topology::container::dynamic::MultilevelHexahedronSetTopologyContainer*	_multilevelTopology;

    Data<bool>		_bRecursive; ///< Use recursive matrix computation

protected:

    // ---------------  Modified method: compute and re-use MBK
    typedef component::solidmechanics::fem::elastic::HexahedralFEMForceFieldAndMass<DataTypes> Inherited;
    typedef typename Inherited::HexahedronInformation HexahedronInformation;
    typedef typename Inherited::Mat33 Mat33;
    typedef typename Inherited::Displacement Displacement;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data<bool> useMBK; ///< if true, compute and use MBK matrix

    /** Matrix-vector product for implicit methods with iterative solvers.
        If the MBK matrix is ill-conditionned, recompute it, and correct it to avoid too small singular values.
    */
    void addMBKdx(const core::MechanicalParams* mparams, core::MultiVecDerivId dfId) override;

    bool matrixIsDirty;                      ///< Matrix \f$ \alpha M + \beta B + \gamma C \f$ needs to be recomputed
    type::vector< ElementMass > mbkMatrix; ///< Matrix \f$ \alpha M + \beta B + \gamma C \f$

protected:
    virtual void computeCorrection( ElementMass& ) {} ///< Limit the conditioning number of each mbkMatrix as defined by maxConditioning (in derived classes).
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_NONUNIFORM_API NonUniformHexahedralFEMForceFieldAndMass<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::solidmechanics::fem::nonuniform
