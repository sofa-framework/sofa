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
#include <sofa/core/trait/DataTypes.h>
#include <sofa/type/MatSym.h>
#include <sofa/type/VoigtNotation.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * Computes the product between a strain-displacement matrix B and a displacement vector v.
 *
 * This function calculates the strain vector ε = B * v where B is represented in Voigt notation.
 * For diagonal terms (normal strains), it performs a simple dot product.
 * For off-diagonal terms (shear strains), it sums the contributions from both partial derivatives.
 *
 * @tparam StrainDisplacement The type of the strain-displacement matrix.
 * @param B The strain-displacement matrix.
 * @param v The displacement vector.
 * @return The strain vector in Voigt notation.
 */
template <class StrainDisplacement>
sofa::type::Vec<StrainDisplacement::nbLines,
                typename StrainDisplacement::Real>
strainDisplacementVectorProduct(
    const StrainDisplacement& B,
    const sofa::type::Vec<StrainDisplacement::nbColumns,
                          typename StrainDisplacement::Real>& v)
{
    static constexpr auto spatial_dimensions = StrainDisplacement::spatial_dimensions;

    sofa::type::Vec<StrainDisplacement::nbLines,
                    typename StrainDisplacement::Real> product;

    for (std::size_t voigtIndex = 0; voigtIndex < StrainDisplacement::nbLines; ++voigtIndex)
    {
        const auto [i, j] = sofa::type::toTensorIndices<spatial_dimensions>(voigtIndex);

        if (i == j)
        {
            for (sofa::Size ne = 0; ne < StrainDisplacement::nbNodesInElement; ++ne)
            {
                product[voigtIndex] += B(voigtIndex, ne * spatial_dimensions + i) * v(ne * spatial_dimensions + i);
            }
        }
        else
        {
            for (sofa::Size ne = 0; ne < StrainDisplacement::nbNodesInElement; ++ne)
            {
                product[voigtIndex] +=
                    B(voigtIndex, ne * spatial_dimensions + i) * v(ne * spatial_dimensions + i) +
                    B(voigtIndex, ne * spatial_dimensions + j) * v(ne * spatial_dimensions + j);
            }
        }
    }

    return product;
}

/**
 * Computes the product between the transpose of a strain-displacement matrix B and a strain vector v.
 *
 * This function calculates the nodal force vector f = B^T * v where v is a strain (or stress) vector in Voigt notation.
 *
 * @tparam StrainDisplacement The type of the strain-displacement matrix.
 * @param B The strain-displacement matrix.
 * @param v The vector to be multiplied by the transpose of B.
 * @return The resulting vector (typically nodal forces).
 */
template <class StrainDisplacement>
sofa::type::Vec<StrainDisplacement::nbColumns,
                typename StrainDisplacement::Real>
strainDisplacementTransposedVectorProduct(
    const StrainDisplacement& B,
    const sofa::type::Vec<StrainDisplacement::nbLines,
                          typename StrainDisplacement::Real>& v)
{
    static constexpr auto spatial_dimensions = StrainDisplacement::spatial_dimensions;

    sofa::type::Vec<StrainDisplacement::nbColumns,
                typename StrainDisplacement::Real> product;

    for (std::size_t ne = 0; ne < StrainDisplacement::nbNodesInElement; ++ne)
    {
        for (std::size_t voigtIndex = 0; voigtIndex < StrainDisplacement::nbLines; ++voigtIndex)
        {
            const auto [i, j] = sofa::type::toTensorIndices<spatial_dimensions>(voigtIndex);

            product[ne * spatial_dimensions + i] +=
                B(voigtIndex, ne * spatial_dimensions + i) * v[voigtIndex];

            if (i != j)
            {
                product[ne * spatial_dimensions + j] +=
                    B(voigtIndex, ne * spatial_dimensions + j) * v[voigtIndex];
            }
        }
    }

    return product;
}

/**
 * Represents the strain-displacement matrix (B-matrix) for a finite element.
 *
 * The B-matrix relates the strain vector (in Voigt notation) to the nodal displacement vector: ε = B * u.
 *
 * @tparam DataTypes The traits defining the simulation's data types (e.g., Vec3d).
 * @param ElementType The type of finite element (e.g., Tetrahedron).
 */
template <class DataTypes, class ElementType>
struct StrainDisplacement
{
    using Real = sofa::Real_t<DataTypes>;
    static constexpr auto spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr auto nbLines = sofa::type::NumberOfIndependentElements<spatial_dimensions>;
    static constexpr auto nbNodesInElement = ElementType::NumberOfNodes;
    static constexpr auto nbColumns = nbNodesInElement * spatial_dimensions;

    /** Accesses the element at (i, j) in the strain-displacement matrix. */
    constexpr Real& operator()(sofa::Size i, sofa::Size j) noexcept
    {
        return B(i, j);
    }

    /** Accesses the element at (i, j) in the strain-displacement matrix (const). */
    constexpr const Real& operator()(sofa::Size i, sofa::Size j) const noexcept
    {
        return B(i, j);
    }

    /**
     * Multiplies the B-matrix by a displacement vector.
     * @param v Displacement vector.
     * @return Resulting strain vector in Voigt notation.
     */
    constexpr sofa::type::Vec<nbLines, Real> operator*(const sofa::type::Vec<nbColumns, Real>& v) const
    {
        return strainDisplacementVectorProduct(*this, v);
    }

    /**
     * Multiplies the transpose of the B-matrix by a matrix.
     * @tparam C Number of columns in the matrix v.
     * @param v Matrix to multiply with B^T.
     * @return Resulting matrix B^T * v.
     */
    template<sofa::Size C>
    constexpr sofa::type::Mat<nbColumns, C, Real> multTranspose(const sofa::type::Mat<nbLines, C, Real>& v) const noexcept
    {
        return B.multTranspose(v);
    }

    /**
     * Multiplies the transpose of the B-matrix by a strain vector.
     * @param v Strain (or stress) vector in Voigt notation.
     * @return Resulting nodal force vector.
     */
    constexpr sofa::type::Vec<nbColumns, Real> multTranspose(const sofa::type::Vec<nbLines, Real>& v) const noexcept
    {
        return strainDisplacementTransposedVectorProduct(*this, v);
    }

    /** Internal representation of the strain-displacement matrix. */
    sofa::type::Mat<nbLines, nbColumns, sofa::Real_t<DataTypes>> B;

    /** Outputs the B-matrix to a stream. */
    friend std::ostream& operator<<(std::ostream& out, const StrainDisplacement& B)
    {
        return operator<<(out, B.B);
    }
};

/**
 * Multiplies a matrix by a strain-displacement matrix.
 * @tparam L Number of lines in matrix A.
 * @tparam C Number of columns in matrix A (must match nbLines of B).
 * @param A Matrix to multiply with B.
 * @param B Strain-displacement matrix.
 * @return Resulting matrix product A * B.
 */
template <sofa::Size L, sofa::Size C, class real, class DataTypes, class ElementType>
sofa::type::Mat<L, StrainDisplacement<DataTypes, ElementType>::nbColumns, real>
operator*(
    const sofa::type::Mat<L, C, real>& A,
    const StrainDisplacement<DataTypes, ElementType>& B)
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
    const sofa::type::Mat<ElementType::NumberOfNodes, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes> >& gradientShapeFunctions)
{
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr auto nbVoigtIndices = sofa::type::NumberOfIndependentElements<DataTypes::spatial_dimensions>;

    StrainDisplacement<DataTypes, ElementType> B;

    for (std::size_t voigtIndex = 0; voigtIndex < nbVoigtIndices; ++voigtIndex)
    {
        const auto [i, j] = sofa::type::toTensorIndices<spatial_dimensions>(voigtIndex);
        for (sofa::Size ne = 0; ne < NumberOfNodesInElement; ++ne)
        {
            B(voigtIndex, ne * spatial_dimensions + i) = gradientShapeFunctions[ne][j];
            if (i != j)
            {
                //engineering shear strain (no factor 1/2)
                B(voigtIndex, ne * spatial_dimensions + j) = gradientShapeFunctions[ne][i];
            }
        }
    }

    return B;
}

}
