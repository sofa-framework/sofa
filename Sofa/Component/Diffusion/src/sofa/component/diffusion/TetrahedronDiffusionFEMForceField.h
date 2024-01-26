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
#include <sofa/component/diffusion/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::component::diffusion
{

template<class DataTypes>
class TetrahedronDiffusionFEMForceField : public core::behavior::ForceField<DataTypes>
{
   public:
      SOFA_CLASS(SOFA_TEMPLATE(TetrahedronDiffusionFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

      typedef core::behavior::ForceField<DataTypes> Inherited;
      typedef typename DataTypes::VecCoord VecCoord;
      typedef typename DataTypes::VecDeriv VecDeriv;
      typedef typename DataTypes::Coord    Coord   ;
      typedef typename DataTypes::Deriv    Deriv   ;
      typedef typename Coord::value_type   Real    ;
      typedef typename sofa::type::vector< Real > VectorReal    ;

      /// assumes the mechanical object type (3D)
      typedef type::Vec<3,Real>                            Vec3;
      typedef defaulttype::StdVectorTypes< Vec3, Vec3, Real >     MechanicalTypes ;
      typedef sofa::core::behavior::MechanicalState<MechanicalTypes>      MechObject;

      typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
      typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

public:
      /// Constructor
      TetrahedronDiffusionFEMForceField();
      /// Destructor
      virtual ~TetrahedronDiffusionFEMForceField();

      //@{
      /** Other usual SOFA functions */
      void init() override;
      void reinit() override;
      void draw(const core::visual::VisualParams*) override;
      //@}

      /// Forcefield functions for Matrix system. Adding force to global forcefield vector.
      void addForce (const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& dF, const DataVecCoord& dX, const DataVecDeriv& /*v*/) override;
      /// Forcefield functions for Matrix system. Adding derivate force to global forcefield vector.
      void addDForce(const sofa::core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& dF , const DataVecDeriv& dX) override;
      /// Forcefield functions for Matrix system. Adding derivate force to global forcefield vector. (direct solver)
      void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

      void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
      void buildDampingMatrix(core::behavior::DampingMatrix* /* matrices */) override {}

      /// Return Potential energy of the mesh.
      SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const override;

      /// Get diffusion coefficient coefficient
      sofa::type::vector<Real> getDiffusionCoefficient();
      /// Get diffusion coefficient for tetra i
      Real getTetraDiffusionCoefficient(Index i);

      /// Set diffusion coefficient with constant value
      void setDiffusionCoefficient(const Real val);
      /// Set diffusion coefficient with vector of value for each tetra
      void setDiffusionCoefficient(const sofa::type::vector<Real> val);





      /// Single value for diffusion coefficient (constant coefficient)
      Data<Real> d_constantDiffusionCoefficient;
      /// Vector of diffusivities associated with all tetras
      Data<sofa::type::vector<Real> > d_tetraDiffusionCoefficient;
      /// bool used to specify 1D diffusion
      /// This data is now useless, as it can be deduced from the template
      DeprecatedAndRemoved d_1DDiffusion;

      /// Ratio for anisotropic diffusion
      Data<Real> d_transverseAnisotropyRatio;
      /// Vector for transverse anisotropy
      Data<sofa::type::vector<Vec3> > d_transverseAnisotropyDirectionArray;
      /// Mechanic xml tags of the system.
      Data<std::string> d_tagMeshMechanics;
      /// Boolean enabling to visualize the different diffusion coefficient
      Data <bool> d_drawConduc;

      /// Link to be set to the topology container in the component graph. 
      SingleLink<TetrahedronDiffusionFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


protected:
      /// Function computing the edge diffusion coefficient from tetrahedral information
      void computeEdgeDiffusionCoefficient();

      /// Vector saving the edge diffusion coefficients
      sofa::type::vector<Real> edgeDiffusionCoefficient;
      /// Pointer to mechanical mechanicalObject
      typename MechObject::SPtr mechanicalObject;
      /// Pointer to topology
      sofa::core::topology::BaseMeshTopology::SPtr m_topology;
      /// Saving the number of edges
      sofa::Size nbEdges;

public:
      /// Boolean if the diffusion coefficients have loaded from file
      bool loadedDiffusivity;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_DIFFUSION_API TetrahedronDiffusionFEMForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_DIFFUSION_API TetrahedronDiffusionFEMForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_DIFFUSION_API TetrahedronDiffusionFEMForceField<defaulttype::Vec3Types>;
 
#endif

} // namespace sofa::component::diffusion
