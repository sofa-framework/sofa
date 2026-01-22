#pragma once

#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/FullySymmetric4Tensor.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/IsotropicElasticityTensor.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/MatrixTools.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/StrainDisplacement.h>
#include <sofa/type/Mat.h>

#include <ostream>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
using ElementStiffness = sofa::type::Mat<
    ElementType::NumberOfNodes * DataTypes::spatial_dimensions,
    ElementType::NumberOfNodes * DataTypes::spatial_dimensions,
    sofa::Real_t<DataTypes>
>;

/**
 * Specifies the type of matrix-vector product to be used with the stiffness matrix.
 */
enum class MatrixVectorProductType
{
    /**
     * The matrix-vector product is computed using the factorization of the matrix
     */
    Factorization,

    /**
     * The matrix-vector product is computed using the dense matrix representation
     */
    Dense
};

/**
 * Represents an element stiffness matrix. It contains the dense matrix representation, but also
 * its factorization. Both the factorization or the dense matrix can be used for the product of the
 * stiffness matrix with a vector. Although it gives the same result, it has an impact on the
 * performances.
 */
template <class DataTypes, class ElementType, MatrixVectorProductType matrixVectorProductType>
struct FactorizedElementStiffness
{
private:
    using Real = sofa::Real_t<DataTypes>;
    using FiniteElement = sofa::component::solidmechanics::fem::elastic::FiniteElement<ElementType, DataTypes>;
    static constexpr auto NbQuadraturePoints = FiniteElement::quadraturePoints().size();
    static constexpr sofa::Size NumberOfIndependentElements = symmetric_tensor::NumberOfIndependentElements<DataTypes::spatial_dimensions>;

    /// The factors of the stiffness matrix
    /// @{
    std::array<StrainDisplacement<DataTypes, ElementType>, NbQuadraturePoints> B;
    IsotropicElasticityTensor<DataTypes> elasticityTensor;
    std::array<Real, NbQuadraturePoints> factors;
    /// @}

    /**
     * The assembled stiffness matrix
     */
    sofa::type::Mat<
        ElementType::NumberOfNodes * DataTypes::spatial_dimensions,
        ElementType::NumberOfNodes * DataTypes::spatial_dimensions,
        sofa::Real_t<DataTypes>> stiffnessMatrix;

public:
    const StrainDisplacement<DataTypes, ElementType>& getB(std::size_t i) const { return B[i]; }
    Real getFactor(std::size_t i) const { return factors[i]; }
    const IsotropicElasticityTensor<DataTypes>& getElasticityTensor() const { return elasticityTensor; }

    void setElasticityTensor(const FullySymmetric4Tensor<DataTypes>& elasticityTensor_)
    {
        for (std::size_t i = 0; i < NumberOfIndependentElements; ++i)
        {
            for (std::size_t j = 0; j < NumberOfIndependentElements; ++j)
            {
                this->elasticityTensor(i, j) = elasticityTensor_.toVoigtMatSym()(i, j);
            }
        }
    }

    void addFactor(std::size_t quadraturePointIndex,
        const Real factor,
        const StrainDisplacement<DataTypes, ElementType>& B_)
    {
        this->factors[quadraturePointIndex] = factor;
        this->B[quadraturePointIndex] = B_;

        const auto K_i = factor * B_.multTranspose(elasticityTensor.toMat() * B_.B);
        stiffnessMatrix += K_i;
    }

    const auto& getAssembledMatrix() const { return stiffnessMatrix; }

    using Vec = sofa::type::Vec<
        ElementType::NumberOfNodes * DataTypes::spatial_dimensions,
        sofa::Real_t<DataTypes>>;

    inline Vec operator*(const Vec& v) const
    {
        if constexpr (matrixVectorProductType == MatrixVectorProductType::Factorization)
        {
            if constexpr (NbQuadraturePoints > 1)
            {
                Vec result;
                for (std::size_t i = 0; i < NbQuadraturePoints; ++i)
                {
                    const auto& B = this->B[i];
                    result += B.multTranspose(this->factors[i] * (this->elasticityTensor * (B * v)));
                }
                return result;
            }
            else
            {
                return this->B[0].multTranspose(this->factors[0] * (this->elasticityTensor * (this->B[0] * v)));
            }
        }
        else
        {
            return this->stiffnessMatrix * v;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const FactorizedElementStiffness& obj)
    {
        return os << obj.getAssembledMatrix();
    }

    friend std::istream& operator>>(std::istream& in, FactorizedElementStiffness& obj)
    {
        return in;
    }
};

template <class DataTypes, class ElementType, MatrixVectorProductType matrixVectorProductType = MatrixVectorProductType::Dense>
FactorizedElementStiffness<DataTypes, ElementType, matrixVectorProductType> integrate(
    const std::array<sofa::Coord_t<DataTypes>, ElementType::NumberOfNodes>& nodesCoordinates,
    const FullySymmetric4Tensor<DataTypes>& elasticityTensor)
{
    using Real = sofa::Real_t<DataTypes>;
    using FiniteElement = FiniteElement<ElementType, DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    FactorizedElementStiffness<DataTypes, ElementType, matrixVectorProductType> K;
    K.setElasticityTensor(elasticityTensor);

    std::size_t quadraturePointIndex = 0;
    for (const auto& [quadraturePoint, weight] : FiniteElement::quadraturePoints())
    {
        // gradient of shape functions in the reference element evaluated at the quadrature point
        const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> dN_dq_ref =
            FiniteElement::gradientShapeFunctions(quadraturePoint);

        // jacobian of the mapping from the reference space to the physical space, evaluated at the
        // quadrature point
        sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real> jacobian;
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
            jacobian += sofa::type::dyad(nodesCoordinates[i], dN_dq_ref[i]);

        const auto detJ = sofa::component::solidmechanics::fem::elastic::absGeneralizedDeterminant(jacobian);
        const sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real> J_inv =
            sofa::component::solidmechanics::fem::elastic::inverse(jacobian);

        // gradient of the shape functions in the physical element evaluated at the quadrature point
        sofa::type::Mat<NumberOfNodesInElement, spatial_dimensions, Real> dN_dq(sofa::type::NOINIT);
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
            dN_dq[i] = J_inv.transposed() * dN_dq_ref[i];

        const auto B = makeStrainDisplacement<DataTypes, ElementType>(dN_dq);
        K.addFactor(quadraturePointIndex, weight * detJ, B);

        ++quadraturePointIndex;
    }
    return K;
}



}
