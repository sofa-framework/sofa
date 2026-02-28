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
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/ComputeStrategy.h>
#include <sofa/core/visual/DrawMesh.h>
#include <sofa/simulation/task/ParallelForEach.h>
#include <sofa/simulation/task/TaskSchedulerUser.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/TopologyAccessor.h>

#if !defined(ELASTICITY_COMPONENT_FEM_FORCEFIELD_CPP)
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes, class ElementType>
class FEMForceField :
    public virtual sofa::core::behavior::ForceField<DataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor,
    public virtual sofa::simulation::TaskSchedulerUser
{
public:
    SOFA_CLASS3(SOFA_TEMPLATE2(FEMForceField, DataTypes, ElementType),
        sofa::core::behavior::ForceField<DataTypes>,
        sofa::core::behavior::TopologyAccessor,
        sofa::simulation::TaskSchedulerUser);

private:
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    using ElementForce = typename trait::ElementForce;

public:
    void init() override;

    void addForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& f,
        const sofa::DataVecCoord_t<DataTypes>& x,
        const sofa::DataVecDeriv_t<DataTypes>& v) override;

    void addDForce(const sofa::core::MechanicalParams* mparams,
                   sofa::DataVecDeriv_t<DataTypes>& df,
                   const sofa::DataVecDeriv_t<DataTypes>& dx) override;

    void draw(const sofa::core::visual::VisualParams*) override;

    sofa::Data<ComputeStrategy> d_computeForceStrategy;
    sofa::Data<ComputeStrategy> d_computeForceDerivStrategy;

    sofa::Data<sofa::Real_t<DataTypes>> d_elementSpace;

protected:

    FEMForceField();

    /// Methods related to addForce
    /// @{
    void computeElementsForces(const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& f,
        const sofa::VecCoord_t<DataTypes>& x);

    virtual void beforeElementForce(const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& f,
        const sofa::VecCoord_t<DataTypes>& x) {}

    virtual void computeElementsForces(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& f,
        const sofa::VecCoord_t<DataTypes>& x) = 0;

    void dispatchElementForcesToNodes(
        const sofa::type::vector<typename trait::TopologyElement>& elements,
        sofa::VecDeriv_t<DataTypes>& nodeForces);
    /// @}


    /// Methods related to addDForce
    /// @{
    void computeElementsForcesDeriv(const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& df,
        const sofa::VecDeriv_t<DataTypes>& dx);

    virtual void computeElementsForcesDeriv(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& df,
        const sofa::VecDeriv_t<DataTypes>& dx) = 0;

    /**
     * Force derivatives were computed at the element level. This function dispatches the force
     * derivatives from the elements to the nodes.
     */
    void dispatchElementForcesDerivToNodes(const sofa::core::MechanicalParams* mparams,
        const sofa::type::vector<typename trait::TopologyElement>& elements,
        sofa::VecDeriv_t<DataTypes>& nodeForcesDeriv);
    /// @}

    sofa::simulation::ForEachExecutionPolicy getExecutionPolicy(const sofa::Data<ComputeStrategy>& strategy) const;

    sofa::type::vector<sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<DataTypes>>> m_elementForce;
    sofa::type::vector<sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<DataTypes>>> m_elementDForce;

    sofa::core::visual::DrawElementMesh<ElementType> m_drawMesh;
};

#if !defined(ELASTICITY_COMPONENT_FEM_FORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
