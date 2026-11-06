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
#include <sofa/component/solidmechanics/fem/elastic/BaseSourceTerm.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/fem/FiniteElement.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_SOURCE_TERM_INTEGRATOR_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class FEMSourceTermIntegrator
 * @brief Integrates source terms into consistent nodal loads,
 * \f$ \int_{\Omega} N_a \, r \, d\Omega \f$, with \f$ N_a \f$ the shape function of node a and r
 * the source density.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 * @tparam TElementType The type of finite element (e.g., sofa::geometry::Tetrahedron).
 */
template <class TDataTypes, class TElementType>
class FEMSourceTermIntegrator :
    public sofa::core::behavior::ForceField<TDataTypes>,
    public virtual sofa::core::behavior::TopologyAccessor
{
public:
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
    SOFA_CLASS2(SOFA_TEMPLATE2(FEMSourceTermIntegrator, DataTypes, ElementType),
        sofa::core::behavior::ForceField<DataTypes>,
        sofa::core::behavior::TopologyAccessor);

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;
    using Real = sofa::Real_t<DataTypes>;

    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

public:

    /**
     * @brief Source terms integrated once on the rest configuration.
     */
    sofa::MultiLink<FEMSourceTermIntegrator<DataTypes, ElementType>, BaseSourceTerm<DataTypes, ElementType>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_constantSources;

    /**
     * @brief Source terms integrated on the current configuration.
     */
    sofa::MultiLink<FEMSourceTermIntegrator<DataTypes, ElementType>, BaseSourceTerm<DataTypes, ElementType>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_nonConstantSources;

    /**
     * @brief Initializes the component.
     */
    void init() override;

    /**
     * @brief Adds the nodal loads of the source terms.
     *
     * @param mparams Mechanical parameters for the computation.
     * @param f The force vector, to which the source term will be added.
     * @param x The current positions.
     * @param v The current velocities.
     */
    void addForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& f,
        const sofa::DataVecCoord_t<DataTypes>& x,
        const sofa::DataVecDeriv_t<DataTypes>& v) override;

    /**
     * @brief Applies the tangent of the non-constant source terms to a displacement increment.
     */
    void addDForce(const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& df,
        const sofa::DataVecDeriv_t<DataTypes>& dx) override;

    /**
     * @brief Assembles the tangent of the non-constant source terms.
     */
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    using sofa::core::behavior::ForceField<DataTypes>::getPotentialEnergy;
    /**
     * @brief Potential energy of the nodal load, \f$ V = -\sum_a F_a \cdot (x_a - x_{0,a}) \f$.
     */
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
        const sofa::DataVecCoord_t<DataTypes>& x) const override;

    /**
     * @brief Degree of the quadrature rule integrating the source terms.
     */
    sofa::Data<sofa::Size> d_quadratureDegree;

    /**
     * @brief Whether the non-symmetric tangent of the non-constant source terms is assembled.
     */
    sofa::Data<bool> d_useTangentStiffness;

protected:

    FEMSourceTermIntegrator();

    /**
     * @brief Ensures that valid source terms are linked, falling back to the current context.
     */
    void validateSources();

    /**
     * @brief Re-integrates the constant terms when a Data they read has changed.
     */
    void doUpdateInternal() override;

    /**
     * @brief Integrates the constant source terms on the rest configuration.
     */
    void assembleConstantForce();

    /**
     * @brief Integrates the non-constant source terms.
     *
     * @param x The configuration the terms are integrated on.
     * @param f The force vector the nodal forces are accumulated into.
     */
    void assembleNonConstantForce(const sofa::VecCoord_t<DataTypes>& x,
        sofa::VecDeriv_t<DataTypes>& f) const;

    /**
     * @brief Nodal load of the constant source terms.
     */
    sofa::VecDeriv_t<DataTypes> m_constantForce;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_SOURCE_TERM_INTEGRATOR_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMSourceTermIntegrator<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
