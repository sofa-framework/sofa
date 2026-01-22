#pragma once
#include <sofa/core/trait/DataTypes.h>
#include <sofa/type/Mat.h>

#include <sofa/component/solidmechanics/fem/elastic/impl/SymmetricTensor.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
struct StrainDisplacement;


template<sofa::Size nbLines, sofa::Size nbColumns, class real>
sofa::type::Vec<nbLines, real> strainDisplacementVectorProduct(const sofa::type::Mat<nbLines, nbColumns, real>& B, const sofa::type::Vec<nbColumns, real>& v)
{
    return B * v;
}

template <sofa::Size nbLines, sofa::Size nbColumns, class real>
sofa::type::Vec<nbColumns, real> strainDisplacementTransposedVectorProduct(
    const sofa::type::Mat<nbLines, nbColumns, real>& B, const sofa::type::Vec<nbLines, real>& v)
{
    return B.multTranspose(v);
}

/**
 * Specialization for a linear tetrahedron where the operations containing known zeros in the strain
 * displacement tensor are omitted.
 */
template <class real>
sofa::type::Vec<6, real> strainDisplacementVectorProduct(const sofa::type::Mat<6, 12, real>& B, const sofa::type::Vec<12, real>& v)
{
    sofa::type::Vec<6, real> Bv{sofa::type::NOINIT};

    Bv[0] = B(0, 0) * v[0] + B(0, 3) * v[3] + B(0, 6) * v[6] + B(0, 9) * v[9];
    Bv[1] = B(1, 1) * v[1] + B(1, 4) * v[4] + B(1, 7) * v[7] + B(1, 10) * v[10];
    Bv[2] = B(2, 2) * v[2] + B(2, 5) * v[5] + B(2, 8) * v[8] + B(2, 11) * v[11];
    Bv[3] = B(3, 0) * v[0] + B(3, 1) * v[1] + B(3, 3) * v[3] + B(3, 4) * v[4] + B(3, 6) * v[6] + B(3, 7) * v[7] + B(3, 9) * v[9] + B(3, 10) * v[10];
    Bv[4] = B(4, 0) * v[0] + B(4, 2) * v[2] + B(4, 3) * v[3] + B(4, 5) * v[5] + B(4, 6) * v[6] + B(4, 8) * v[8] + B(4, 9) * v[9] + B(4, 11) * v[11];
    Bv[5] = B(5, 1) * v[1] + B(5, 2) * v[2] + B(5, 4) * v[4] + B(5, 5) * v[5] + B(5, 7) * v[7] + B(5, 8) * v[8] + B(5, 10) * v[10] + B(5, 11) * v[11];

    return Bv;
}

/**
 * Specialization for a linear tetrahedron where the operations containing known zeros in the strain
 * displacement tensor are omitted.
 */
template<class real>
sofa::type::Vec<12, real> strainDisplacementTransposedVectorProduct(const sofa::type::Mat<6, 12, real>& B, const sofa::type::Vec<6, real>& v)
{
    sofa::type::Vec<12, real> B_Tv { sofa::type::NOINIT };

    B_Tv[0] = B(0, 0) * v[0] + B(3, 0) * v[3] + B(4, 0) * v[4];
    B_Tv[1] = B(1, 1) * v[1] + B(3, 1) * v[3] + B(5, 1) * v[5];
    B_Tv[2] = B(2, 2) * v[2] + B(4, 2) * v[4] + B(5, 2) * v[5];
    B_Tv[3] = B(0, 3) * v[0] + B(3, 3) * v[3] + B(4, 3) * v[4];
    B_Tv[4] = B(1, 4) * v[1] + B(3, 4) * v[3] + B(5, 4) * v[5];
    B_Tv[5] = B(2, 5) * v[2] + B(4, 5) * v[4] + B(5, 5) * v[5];
    B_Tv[6] = B(0, 6) * v[0] + B(3, 6) * v[3] + B(4, 6) * v[4];
    B_Tv[7] = B(1, 7) * v[1] + B(3, 7) * v[3] + B(5, 7) * v[5];
    B_Tv[8] = B(2, 8) * v[2] + B(4, 8) * v[4] + B(5, 8) * v[5];
    B_Tv[9] = B(0, 9) * v[0] + B(3, 9) * v[3] + B(4, 9) * v[4];
    B_Tv[10] = B(1, 10) * v[1] + B(3, 10) * v[3] + B(5, 10) * v[5];
    B_Tv[11] = B(2, 11) * v[2] + B(4, 11) * v[4] + B(5, 11) * v[5];

    return B_Tv;
}


template <class DataTypes, class ElementType>
struct StrainDisplacement
{
    using Real = sofa::Real_t<DataTypes>;
    static constexpr auto nbLines = symmetric_tensor::NumberOfIndependentElements<DataTypes::spatial_dimensions>;
    static constexpr auto nbColumns = ElementType::NumberOfNodes * DataTypes::spatial_dimensions;

    constexpr Real& operator()(sofa::Size i, sofa::Size j) noexcept
    {
        return B(i, j);
    }

    constexpr const Real& operator()(sofa::Size i, sofa::Size j) const noexcept
    {
        return B(i, j);
    }

    constexpr sofa::type::Vec<nbLines, Real> operator*(const sofa::type::Vec<nbColumns, Real>& v) const
    {
        return strainDisplacementVectorProduct(B, v);
    }

    template<sofa::Size C>
    constexpr sofa::type::Mat<nbColumns, C, Real> multTranspose(const sofa::type::Mat<nbLines, C, Real>& v) const noexcept
    {
        return B.multTranspose(v);
    }

    constexpr sofa::type::Vec<nbColumns, Real> multTranspose(const sofa::type::Vec<nbLines, Real>& v) const noexcept
    {
        return strainDisplacementTransposedVectorProduct(B, v);
    }

    sofa::type::Mat<nbLines, nbColumns, sofa::Real_t<DataTypes>> B;

    friend std::ostream& operator<<(std::ostream& out, const StrainDisplacement& B)
    {
        return operator<<(out, B.B);
    }
};

template <sofa::Size L, sofa::Size C, class real, class DataTypes, class ElementType>
sofa::type::Mat<L, StrainDisplacement<DataTypes, ElementType>::nbColumns, real>
operator*(const sofa::type::Mat<L, C, real>& A, const StrainDisplacement<DataTypes, ElementType>& B)
{
    return A * B.B;
}

/**
 * Creates a strain-displacement matrix (B-matrix) for finite element calculations.
 *
 * This function constructs a strain-displacement matrix based on the provided gradient of shape
 * functions. The matrix is filled according to spatial dimensions and the number of nodes in the
 * element.
 *
 * @param gradientShapeFunctions A matrix containing the gradient of the shape functions.
 *
 * @return A strain-displacement matrix of type StrainDisplacement<DataTypes, ElementType>,
 * constructed for the finite element with the given gradient shape functions.
 */
template <class DataTypes, class ElementType>
StrainDisplacement<DataTypes, ElementType> makeStrainDisplacement(
    const sofa::type::Mat<ElementType::NumberOfNodes, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes> > gradientShapeFunctions)
{
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;

    StrainDisplacement<DataTypes, ElementType> B;
    for (sofa::Size ne = 0; ne < NumberOfNodesInElement; ++ne)
    {
        for (sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            B(i, ne * spatial_dimensions + i) = gradientShapeFunctions[ne][i];
        }

        auto row = spatial_dimensions;
        for (sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for (sofa::Size j = i + 1; j < spatial_dimensions; ++j)
            {
                B(row, ne * spatial_dimensions + i) = gradientShapeFunctions[ne][j];
                B(row, ne * spatial_dimensions + j) = gradientShapeFunctions[ne][i];
                ++row;
            }
        }
    }

    return B;
}

}
