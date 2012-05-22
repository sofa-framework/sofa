/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_POLARDECOMPOSE_H
#define SOFA_HELPER_POLARDECOMPOSE_H


#include <sofa/helper/helper.h>

#include <sofa/defaulttype/Mat.h>


namespace sofa
{

namespace helper
{

template<class Real>
class SOFA_HELPER_API Decompose
{

public:

    /** @name QR
      * @{
      */

    /** QR decomposition
      Compute an orthonormal right-handed 3x3 basis based on two vectors using Gram-Schmidt orthogonalization.
      The basis vectors are the columns of the matrix R. The matrix represents the rotation of the local frame with respect to the reference frame.
      The first basis vector is aligned to the first given vector, the second basis vector is in the plane of the two first given vectors, and the third basis vector is orthogonal to the two others.
      Undefined result if one of the vectors is null, or if the two vectors are parallel.
      */
    static void getRotation( defaulttype::Mat<3,3,Real>& r, defaulttype::Vec<3,Real>& edgex, defaulttype::Vec<3,Real>& edgey );

    /** QR decomposition
      Compute an orthonormal right-handed 3x3 basis based on a matrix using Gram-Schmidt orthogonalization.
      The basis vectors are the columns of the matrix R. The matrix represents the rotation of the local frame with respect to the reference frame.
      The first basis vector is aligned to the first given vector, the second basis vector is in the plane of the two first given vectors, and the third basis vector is orthogonal to the two others.
      Undefined result if one of the vectors is null, or if the two vectors are parallel.
      */
    static void QRDecomposition( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &R );

    /** QR decomposition stable to null columns.
      * Result is still undefined if two columns are parallel.
      * In the clean case (not degenerated), there are only two additional 'if(x<e)' but no additional computations.
      */
    static void QRDecomposition_stable( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &R );







    /** @}
      * @name Polar
      * @{
      */


    /** Polar Decomposition of 3x3 matrix,
     * M = QS.  See Nicholas Higham and Robert S. Schreiber,
     * Fast Polar Decomposition of An Arbitrary Matrix,
     * Technical Report 88-942, October 1988,
     * Department of Computer Science, Cornell University.
     */
    static Real polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q, defaulttype::Mat<3,3,Real>& S );

    /** The same than previous except we do not care about S
     */
    static Real polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q );

    /** Polar decomposition of a 2x2 matix M = QS
     *  Analytic formulation given in
     *  "Matrix Animation and Polar Decomposition"
     *  Ken Shoemake, Computer Graphics Laboratory, University of Pennsylvania
     *  Tom Duff, AT&T Bell Laboratories, Murray Hill
     */
    static void polarDecomposition( const defaulttype::Mat<2,2,Real>& M, defaulttype::Mat<2,2,Real>& Q );






    /** @}
      * @name SVD
      * @{
      */

    /** Non-iterative & faster Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 3x3 Real Matrix @param M
      * Derived from Wild Magic Library
      */
    static void eigenDecomposition( const defaulttype::Mat<3,3,Real> &A, defaulttype::Mat<3,3,Real> &V, defaulttype::Vec<3,Real> &diag );

    /// Non-iterative Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 2x2 Real Matrix @param M
    static void eigenDecomposition( const defaulttype::Mat<2,2,Real> &A, defaulttype::Mat<2,2,Real> &V, defaulttype::Vec<2,Real> &diag );


    /** @} */




private:



    /// threshold for zero comparison (1e-6 for float and 1e-8 for double)
    static Real zeroTolerance();




    /** @internal useful for polarDecomposition
      * Set MadjT to transpose of inverse of M times determinant of M
      */
    static void adjoint_transpose(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT);

    /** @internal useful for polarDecomposition
      * Compute the infinity norm of M
      */
    static Real norm_inf(const defaulttype::Mat<3,3,Real>& M);

    /** @internal useful for polarDecomposition
      * Compute the 1 norm of M
      */
    static Real norm_one(const defaulttype::Mat<3,3,Real>& M);

    /** @internal useful for polarDecomposition
      * Return index of column of M containing maximum abs entry, or -1 if M=0
      */
    static int find_max_col(const defaulttype::Mat<3,3,Real>& M);

    /** @internal useful for polarDecomposition
      * Setup u for Household reflection to zero all v components but first
      */
    static void make_reflector(const defaulttype::Vec<3,Real>& v, defaulttype::Vec<3,Real>& u);

    /** @internal useful for polarDecomposition
      * Apply Householder reflection represented by u to column vectors of M
      */
    static void reflect_cols(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u);
    /** @internal useful for polarDecomposition
      * Apply Householder reflection represented by u to row vectors of M
      */
    static void reflect_rows(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u);

    /** @internal useful for polarDecomposition
      * Find orthogonal factor Q of rank 1 (or less) M
      */
    static void do_rank1(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q);

    /** @internal useful for polarDecomposition
      * Find orthogonal factor Q of rank 2 (or less) M using adjoint transpose
      */
    static void do_rank2(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT, defaulttype::Mat<3,3,Real>& Q);




    /// @internal useful for eigenDecomposition
    static void ComputeRoots( const defaulttype::Mat<3,3,Real>& A, double root[3] );

    /// @internal useful for eigenDecomposition
    static bool PositiveRank( defaulttype::Mat<3,3,Real>& M, Real& maxEntry, defaulttype::Vec<3,Real>& maxRow );

    /// @internal useful for eigenDecomposition
    // Input vec0 must be a unit-length vector.  The output vectors
    // {vec0,vec1} are unit length and mutually perpendicular, and
    // {vec0,vec1,vec2} is an orthonormal basis.
    static void GenerateComplementBasis( defaulttype::Vec<3,Real>& vec0, defaulttype::Vec<3,Real>& vec1, const defaulttype::Vec<3,Real>& vec2 );

    /// @internal useful for eigenDecomposition
    static void ComputeVectors( const defaulttype::Mat<3,3,Real>& A, defaulttype::Vec<3,Real>& U2, int i0, int i1, int i2, defaulttype::Mat<3,3,Real> &V, defaulttype::Vec<3,Real> &diag );


}; // class Decompose

} // namespace helper

} // namespace sofa

#endif
