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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_H
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_H
#include "config.h"


#include "HexahedralFEMForceField.h"
#include <sofa/core/behavior/Mass.h>

#include <SofaBaseTopology/TopologyData.h>

namespace sofa
{

namespace component
{

namespace forcefield
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
    typedef typename HexahedralFEMForceFieldT::VecElementStiffness VecElementMass;
    typedef typename HexahedralFEMForceFieldT::ElementStiffness ElementMass;
    typedef core::topology::BaseMeshTopology::index_type Index;
    typedef typename HexahedralFEMForceFieldT::HexahedronInformation HexahedronInformation;
    typedef typename HexahedralFEMForceFieldT::ElementStiffness ElementStiffness;
    typedef typename HexahedralFEMForceFieldT::Element Element;


protected:
    HexahedralFEMForceFieldAndMass();
public:
    virtual void init( );
    virtual void reinit( );

    // -- Mass interface
    virtual  void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

    ///// WARNING this method only add diagonal elements in the given matrix !
    virtual void addMToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    using HexahedralFEMForceFieldT::addKToMatrix;
    using MassT::addKToMatrix;
    ///// WARNING this method only add diagonal elements in the given matrix !
    virtual void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    ///// WARNING this method only add diagonal elements in the given matrix !
    virtual void addMBKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    virtual  void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

    virtual  void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual SReal getKineticEnergy(const core::MechanicalParams* /* mparams */, const DataVecDeriv& /*v*/)  const ///< vMv/2 using dof->getV()
    {serr<<"HexahedralFEMForceFieldAndMass<DataTypes>::getKineticEnergy not yet implemented"<<sendl; return 0;}

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx);

    virtual void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v);

    virtual void draw(const core::visual::VisualParams* vparams);

    SReal getElementMass(unsigned int index) const;

    void setDensity(Real d) {_density.setValue( d );}
    Real getDensity() {return _density.getValue();}




protected:
    virtual void computeElementMasses( ); ///< compute the mass matrices
    Real integrateVolume( int signx, int signy, int signz, Real l0, Real l1, Real l2 );
    virtual void computeElementMass( ElementMass &Mass, Real& totalMass, const helper::fixed_array<Coord,8> &nodes); ///< compute the mass matrix of an element

    void computeParticleMasses();

    void computeLumpedMasses();

protected:
    //HFFHexahedronHandler* hexahedronHandler;

    Data<Real> _density;
    Data<bool> _useLumpedMass;

    topology::HexahedronData<sofa::helper::vector<ElementMass> > _elementMasses; ///< mass matrices per element
    topology::HexahedronData<sofa::helper::vector<Real> > _elementTotalMass; ///< total mass per element

    topology::PointData<sofa::helper::vector<Real> > _particleMasses; ///< masses per particle in order to compute gravity
    topology::PointData<sofa::helper::vector<Coord> > _lumpedMasses; ///< masses per particle computed by lumping mass matrices
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_SIMPLE_FEM_API HexahedralFEMForceFieldAndMass<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_SIMPLE_FEM_API HexahedralFEMForceFieldAndMass<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_H
