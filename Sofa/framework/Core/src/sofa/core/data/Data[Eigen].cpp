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
#include <sofa/core/data/Data[Eigen].h>

#include <sstream>

namespace sofa::core::objectmodel
{

/// Shared IO implementation for Eigen matrices, partially specialized on the
/// whole Eigen::Matrix<Scalar, Rows, Cols, ...> family (dynamic or fixed size,
/// any scalar type) so that a single implementation serves every matrix type.
template<typename MatrixType> struct EigenMatrixDataIO; // only the Eigen::Matrix family is specialized

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct EigenMatrixDataIO<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
{
    using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    /// Reads the dimensions, resizes dynamic matrices accordingly, then reads
    /// the coefficients in row-major order. Sets failbit and returns false on
    /// invalid dimensions (negative, or conflicting with a compile-time size)
    /// or on parsing failure.
    static bool read(std::istream& in, Matrix& matrix)
    {
        Eigen::Index rows = 0;
        Eigen::Index cols = 0;
        in >> rows >> cols;

        const bool dimsValid = !in.fail()
                               && rows >= 0 && cols >= 0
                               && (Rows == Eigen::Dynamic || rows == Rows)
                               && (Cols == Eigen::Dynamic || cols == Cols);
        if (!dimsValid)
        {
            in.setstate(std::ios::failbit);
            return false;
        }

        if constexpr (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic)
        {
            matrix.resize(rows, cols);
        }

        for (Eigen::Index row = 0; row < rows; ++row)
        {
            for (Eigen::Index col = 0; col < cols; ++col)
            {
                in >> matrix(row, col);
            }
        }
        return !in.fail();
    }

    /// Prints the dimensions, then the coefficients in row-major order.
    static void print(std::ostream& out, const Matrix& matrix)
    {
        out << matrix.rows() << ' ' << matrix.cols();
        for (Eigen::Index row = 0; row < matrix.rows(); ++row)
        {
            for (Eigen::Index col = 0; col < matrix.cols(); ++col)
            {
                out << ' ' << matrix(row, col);
            }
        }
    }
};

/// The read() and getValueString() specializations are thin forwarders to the
/// shared EigenMatrixDataIO helper (see Data[Eigen].h).
template<>
bool SOFA_CORE_API Data<Eigen::MatrixXf>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::MatrixXf>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::MatrixXf>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::MatrixXf>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::MatrixXd>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::MatrixXd>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::MatrixXd>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::MatrixXd>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::Matrix2f>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::Matrix2f>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::Matrix2f>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::Matrix2f>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::Matrix2d>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::Matrix2d>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::Matrix2d>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::Matrix2d>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::Matrix3f>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::Matrix3f>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::Matrix3f>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::Matrix3f>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::Matrix3d>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::Matrix3d>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::Matrix3d>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::Matrix3d>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::Matrix4f>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::Matrix4f>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::Matrix4f>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::Matrix4f>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::Matrix4d>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::Matrix4d>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::Matrix4d>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::Matrix4d>::print( out, getValue() );
    return out.str();
}

template<>
bool SOFA_CORE_API Data<Eigen::MatrixXi>::read( const std::string& str )
{
    std::istringstream in( str );
    const bool ok = EigenMatrixDataIO<Eigen::MatrixXi>::read( in, *beginEdit() );
    endEdit();
    return ok;
}

template<>
std::string SOFA_CORE_API Data<Eigen::MatrixXi>::getValueString() const
{
    std::ostringstream out;
    EigenMatrixDataIO<Eigen::MatrixXi>::print( out, getValue() );
    return out.str();
}

} // namespace sofa::core::objectmodel
