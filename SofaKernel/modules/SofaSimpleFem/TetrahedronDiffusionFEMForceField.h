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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseTopology/TopologySubsetData.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/BaseMechanicalState.h>



//FOR the graph :
#include <sofa/core/visual/VisualModel.h>
#include <map>
#include <sofa/helper/map.h>

//FOR the timer
#include <sofa/helper/system/thread/CTime.h>


namespace sofa
{

namespace component
{


namespace forcefield
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using namespace sofa::component::topology;
using namespace sofa::helper::system::thread;


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
      typedef typename sofa::helper::vector< Real > VectorReal    ;

      /// assumes the mechanical object type (3D)
      typedef Vec<3,Real>                            Vec3;
      typedef StdVectorTypes< Vec3, Vec3, Real >     MechanicalTypes ;
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
      virtual void init() override;
      virtual void reinit() override;
      virtual void draw(const core::visual::VisualParams*) override;
      //@}

      /// Forcefield functions for Matrix system. Adding force to global forcefield vector.
      virtual void addForce (const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& dF, const DataVecCoord& dX, const DataVecDeriv& /*v*/) override;
      /// Forcefield functions for Matrix system. Adding derivate force to global forcefield vector.
      virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& dF , const DataVecDeriv& dX) override;
      /// Forcefield functions for Matrix system. Adding derivate force to global forcefield vector. (direct solver)
      void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
      /// Return Potential energy of the mesh.
      virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const override;

      /// Get diffusion coefficient coefficient
      sofa::helper::vector<Real> getDiffusionCoefficient();
      /// Get diffusion coefficient for tetra i
      Real getTetraDiffusionCoefficient(unsigned int i);

      /// Set diffusion coefficient with constant value
      void setDiffusionCoefficient(const Real val);
      /// Set diffusion coefficient with vector of value for each tetra
      void setDiffusionCoefficient(const sofa::helper::vector<Real> val);





      /// Single value for diffusion coefficient (constant coefficient)
      Data<Real> d_constantDiffusionCoefficient;
      /// Vector of diffusivities associated to all tetras
      Data<sofa::helper::vector<Real> > d_tetraDiffusionCoefficient;
      /// bool used to specify 1D diffusion
      Data<bool> d_1DDiffusion;
      /// Ratio for anisotropic diffusion
      Data<Real> d_transverseAnisotropyRatio;
      /// Vector for transverse anisotropy
      Data<sofa::helper::vector<Vec3> > d_transverseAnisotropyDirectionArray;
      /// Mechanic xml tags of the system.
      Data<std::string> d_tagMeshMechanics;
      /// Boolean enabling to visualize the different diffusion coefficient
      Data <bool> d_drawConduc;



protected:
      /// Function computing the edge diffusion coefficient from tetrahedral information
      void computeEdgeDiffusionCoefficient();

      /// Vector saving the edge diffusion coefficients
      sofa::helper::vector<Real> edgeDiffusionCoefficient;
      /// Pointer to mechanical mechanicalObject
      typename MechObject::SPtr mechanicalObject;
      /// Pointer to topology
      sofa::core::topology::BaseMeshTopology::SPtr topology;
      /// Saving the number of edges
      unsigned int nbEdges;

public:
      /// Boolean if the diffusion coefficients have loaded from file
      bool loadedDiffusivity;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec1dTypes>;
extern template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec2dTypes>;
extern template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec1fTypes>;
extern template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec2fTypes>;
extern template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //namespace forcefield

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_H */
