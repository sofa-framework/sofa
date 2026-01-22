#pragma once

#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/HexahedronRotation.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/IdentityRotation.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/PolarDecomposition.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/RotationMethodsContainer.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/StablePolarDecomposition.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/TriangleRotation.h>
#include <sofa/core/behavior/ForceField.h>

#if !defined(ELASTICITY_COMPONENT_ELEMENT_COROTATIONAL_FEM_FORCE_FIELD_CPP)
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @brief A container for rotation computation methods in corotational formulations
 *
 * This class provides element-specific rotation computation strategies for common element types.
 *
 * @tparam DataTypes The data type used throughout the simulation (e.g., sofa::defaulttype::Vec3Types for 3D)
 * @tparam ElementType The element type (e.g., sofa::geometry::Triangle, sofa::geometry::Tetrahedron)
 *
 * Inherits from RotationMethodsContainer with pre-defined rotation strategies.
 *
 * The class is specialized for some elements because some rotation strategies can be used only
 * for specific elements.
 */
template <class DataTypes, class ElementType>
struct RotationMethods : RotationMethodsContainer<DataTypes, ElementType,
    StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation
>
{
    using Inherit = RotationMethodsContainer<DataTypes, ElementType, StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation>;
    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {}
};

template <class Real>
using Vec3Real = sofa::defaulttype::StdVectorTypes<sofa::type::Vec<3, Real>, sofa::type::Vec<3, Real>, Real>;

//partial specialization for linear triangle in 3D
template <class Real>
struct RotationMethods<Vec3Real<Real>, sofa::geometry::Triangle> : RotationMethodsContainer<Vec3Real<Real>, sofa::geometry::Triangle,
    StablePolarDecomposition<Vec3Real<Real>>, PolarDecomposition<Vec3Real<Real>>, IdentityRotation, TriangleRotation<Vec3Real<Real>>
>
{
    using Inherit = RotationMethodsContainer<Vec3Real<Real>, sofa::geometry::Triangle,
        StablePolarDecomposition<Vec3Real<Real>>, PolarDecomposition<Vec3Real<Real>>, IdentityRotation, TriangleRotation<Vec3Real<Real>> >;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(TriangleRotation<Vec3Real<Real>>::getItem().key);
    }
};

//partial specialization for linear tetrahedron
template <class DataTypes>
struct RotationMethods<DataTypes, sofa::geometry::Tetrahedron> : RotationMethodsContainer<DataTypes, sofa::geometry::Tetrahedron,
    StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, TriangleRotation<DataTypes>
>
{
    using Inherit = RotationMethodsContainer<DataTypes, sofa::geometry::Tetrahedron,
        StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, TriangleRotation<DataTypes> >;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(TriangleRotation<DataTypes>::getItem().key);
    }
};

//partial specialization for linear hexahedron
template <class DataTypes>
struct RotationMethods<DataTypes, sofa::geometry::Hexahedron> : RotationMethodsContainer<DataTypes, sofa::geometry::Hexahedron,
    StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, HexahedronRotation<DataTypes>
>
{
    using Inherit = RotationMethodsContainer<DataTypes, sofa::geometry::Hexahedron,
        StablePolarDecomposition<DataTypes>, PolarDecomposition<DataTypes>, IdentityRotation, HexahedronRotation<DataTypes> >;

    explicit RotationMethods(sofa::core::objectmodel::BaseObject* parent) : Inherit(parent)
    {
        this->d_rotationMethod.setValue(HexahedronRotation<DataTypes>::getItem().key);
    }
};


template <class DataTypes, class ElementType>
class ElementCorotationalFEMForceField :
    public BaseElementLinearFEMForceField<DataTypes, ElementType>,
    public FEMForceField<DataTypes, ElementType>
{
public:
    SOFA_CLASS2(
        SOFA_TEMPLATE2(ElementCorotationalFEMForceField, DataTypes, ElementType),
        SOFA_TEMPLATE2(BaseElementLinearFEMForceField, DataTypes, ElementType),
        SOFA_TEMPLATE2(FEMForceField, DataTypes, ElementType));

    /**
     * The purpose of this function is to register the name of this class according to the provided
     * pattern.
     *
     * Example: ElementCorotationalFEMForceField<Vec3Types, sofa::geometry::Edge> will produce
     * the class name "EdgeCorotationalFEMForceField".
     */
    static const std::string GetCustomClassName()
    {
        return std::string(sofa::geometry::elementTypeToString(ElementType::Element_type)) +
               "CorotationalFEMForceField";
    }

    static const std::string GetCustomTemplateName() { return DataTypes::Name(); }

private:
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    using ElementForce = trait::ElementForce;
    using RotationMatrix = sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, sofa::Real_t<DataTypes>>;


public:

    ElementCorotationalFEMForceField();

    void init() override;

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const sofa::DataVecCoord_t<DataTypes>& x) const override;

    const sofa::type::vector<RotationMatrix>& getElementRotations() const { return m_rotations; }

protected:

    void beforeElementForce(const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& f,
        const sofa::VecCoord_t<DataTypes>& x) override;

    void computeElementsForces(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& f,
        const sofa::VecCoord_t<DataTypes>& x) override;

    void computeElementsForcesDeriv(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& elementForcesDeriv,
        const sofa::VecDeriv_t<DataTypes>& nodeDx) override;

    sofa::type::vector<RotationMatrix> m_rotations;
    sofa::type::vector<RotationMatrix> m_initialRotationsTransposed;

    sofa::Coord_t<DataTypes> translation(const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes) const;
    static sofa::Coord_t<DataTypes> computeCentroid(const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes);

    RotationMethods<DataTypes, ElementType> m_rotationMethods;

    void computeRotations(sofa::type::vector<RotationMatrix>& rotations,
        const sofa::VecCoord_t<DataTypes>& nodePositions,
        const sofa::VecCoord_t<DataTypes>& nodeRestPositions);
    void computeInitialRotations();
};



#if !defined(ELASTICITY_COMPONENT_ELEMENT_COROTATIONAL_FEM_FORCE_FIELD_CPP)
// extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
