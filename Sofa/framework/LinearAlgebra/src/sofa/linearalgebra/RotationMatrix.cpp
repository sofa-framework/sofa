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
#define SOFA_SOFABASELINEARSOLVER_ROTATIONMATRIX_DEFINITION 1
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>

namespace sofa::linearalgebra
{

template<class Real>
sofa::SignedIndex RotationMatrix<Real>::rowSize(void) const
{
    return (sofa::SignedIndex)data.size()/3;
}

/// Number of columns
template<class Real>
sofa::SignedIndex RotationMatrix<Real>::colSize(void) const
{
    return (sofa::SignedIndex)data.size()/3;
}

/// Read the value of the element at row i, column j (using 0-based indices)
template<class Real>
SReal RotationMatrix<Real>::element(sofa::SignedIndex i, sofa::SignedIndex j) const
{
    const sofa::SignedIndex bd = j-(i/3)*3;
    if ((bd<0) || (bd>2)) return 0.0 ;

    return (SReal)data[i*3+bd];
}

/// Resize the matrix and reset all values to 0
template<class Real>
void RotationMatrix<Real>::resize(sofa::SignedIndex nbRow, sofa::SignedIndex nbCol)
{
    if (nbRow!=nbCol) return;
    data.resize(nbRow*3);
}

/// Reset all values to 0
template<class Real>
void RotationMatrix<Real>::clear()
{
    data.clear();
}

template<class Real>
void RotationMatrix<Real>::setIdentity()
{
    memset(&data[0],0,data.size());
    for (unsigned j=0;j<data.size();j+=9) {
        for (int i=0;i<3;i++) {
            data[j+i*3+i] = 1.0;
        }
    }
}

/// Write the value of the element at row i, column j (using 0-based indices)
template<class Real>
void RotationMatrix<Real>::set(sofa::SignedIndex i, sofa::SignedIndex j, double v)
{
    const sofa::SignedIndex bd = (i/3)*3;
    if ((j<bd) || (j>bd+2)) return;
    data[i*3+j-bd] = (Real)v;
}

/// Add v to the existing value of the element at row i, column j (using 0-based indices)
template<class Real>
void RotationMatrix<Real>::add(sofa::SignedIndex i, sofa::SignedIndex j, double v)
{
    const sofa::SignedIndex bd = (i/3)*3;
    if ((j<bd) || (j>bd+2)) return;

    data[i*3+j-bd] += (Real)v;
}

template<class Real>
type::vector<Real>& RotationMatrix<Real>::getVector()
{
    return data;
}

template<class Real>
void RotationMatrix<Real>::opMulV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const
{
    //Solve lv = R * lvR
    std::size_t k = 0;
    sofa::SignedIndex l = 0;
    while (k < data.size())
    {
        result->set(l+0,(SReal)(data[k + 0] * v->element(l+0) + data[k + 1] * v->element(l+1) + data[k + 2] * v->element(l+2)));
        result->set(l+1,(SReal)(data[k + 3] * v->element(l+0) + data[k + 4] * v->element(l+1) + data[k + 5] * v->element(l+2)));
        result->set(l+2,(SReal)(data[k + 6] * v->element(l+0) + data[k + 7] * v->element(l+1) + data[k + 8] * v->element(l+2)));
        l+=3;
        k+=9;
    }
}

template<class Real>
void RotationMatrix<Real>::opMulTV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const
{
    std::size_t k = 0;
    sofa::SignedIndex l = 0;
    while (k < data.size())
    {
        result->set(l+0,(SReal)(data[k + 0] * v->element(l+0) + data[k + 3] * v->element(l+1) + data[k + 6] * v->element(l+2)));
        result->set(l+1,(SReal)(data[k + 1] * v->element(l+0) + data[k + 4] * v->element(l+1) + data[k + 7] * v->element(l+2)));
        result->set(l+2,(SReal)(data[k + 2] * v->element(l+0) + data[k + 5] * v->element(l+1) + data[k + 8] * v->element(l+2)));
        l+=3;
        k+=9;
    }
}

/// multiply the transpose current matrix by m matrix and strore the result in m
template<class Real>
void RotationMatrix<Real>::opMulTM(linearalgebra::BaseMatrix * bresult,linearalgebra::BaseMatrix * bm) const
{
    if (RotationMatrix<Real> * m = dynamic_cast<RotationMatrix<Real> * >(bm))
    {
        if (RotationMatrix<Real> * result = dynamic_cast<RotationMatrix<Real> * >(bresult))
        {
            Real tmp[9];
            std::size_t datSz = data.size() < m->data.size() ? data.size() : m->data.size();
            std::size_t minSz = datSz < result->data.size() ? datSz : result->data.size();

            for (std::size_t i=0; i<minSz; i+=9)
            {
                tmp[0] = data[i+0] * m->data[i+0] + data[i+1] * m->data[i+1] + data[i+2] * m->data[i+2];
                tmp[1] = data[i+0] * m->data[i+3] + data[i+1] * m->data[i+4] + data[i+2] * m->data[i+5];
                tmp[2] = data[i+0] * m->data[i+6] + data[i+1] * m->data[i+7] + data[i+2] * m->data[i+8];

                tmp[3] = data[i+3] * m->data[i+0] + data[i+4] * m->data[i+1] + data[i+5] * m->data[i+2];
                tmp[4] = data[i+3] * m->data[i+3] + data[i+4] * m->data[i+4] + data[i+5] * m->data[i+5];
                tmp[5] = data[i+3] * m->data[i+6] + data[i+4] * m->data[i+7] + data[i+5] * m->data[i+8];

                tmp[6] = data[i+6] * m->data[i+0] + data[i+7] * m->data[i+1] + data[i+8] * m->data[i+2];
                tmp[7] = data[i+6] * m->data[i+3] + data[i+7] * m->data[i+4] + data[i+8] * m->data[i+5];
                tmp[8] = data[i+6] * m->data[i+6] + data[i+7] * m->data[i+7] + data[i+8] * m->data[i+8];

                result->data[i+0] = tmp[0]; result->data[i+1] = tmp[1]; result->data[i+2] = tmp[2];
                result->data[i+3] = tmp[3]; result->data[i+4] = tmp[4]; result->data[i+5] = tmp[5];
                result->data[i+6] = tmp[6]; result->data[i+7] = tmp[7]; result->data[i+8] = tmp[8];
            }

            if (minSz < result->data.size())
            {
                if (datSz<data.size())
                {
                    for (std::size_t i=minSz; i<data.size(); i+=9)
                    {
                        result->data[i+0] = data[i+0]; result->data[i+1] = data[i+1]; result->data[i+2] = data[i+2];
                        result->data[i+3] = data[i+3]; result->data[i+4] = data[i+4]; result->data[i+5] = data[i+5];
                        result->data[i+6] = data[i+6]; result->data[i+7] = data[i+7]; result->data[i+8] = data[i+8];
                    }
                    minSz = data.size();
                }
                else if (datSz<m->data.size())
                {
                    for (std::size_t i=datSz; i<m->data.size(); i+=9)
                    {
                        result->data[i+0] = m->data[i+0]; result->data[i+1] = m->data[i+1]; result->data[i+2] = m->data[i+2];
                        result->data[i+3] = m->data[i+3]; result->data[i+4] = m->data[i+4]; result->data[i+5] = m->data[i+5];
                        result->data[i+6] = m->data[i+6]; result->data[i+7] = m->data[i+7]; result->data[i+8] = m->data[i+8];
                    }
                    minSz = m->data.size();
                }
            }

            if (minSz < result->data.size())
            {
                for (std::size_t i=datSz; i<result->data.size(); i+=9)
                {
                    result->data[i+0] = 1; result->data[i+1] = 0; result->data[i+2] = 0;
                    result->data[i+3] = 0; result->data[i+4] = 1; result->data[i+5] = 0;
                    result->data[i+6] = 0; result->data[i+7] = 0; result->data[i+8] = 1;
                }
            }

            return;
        }
    }
    linearalgebra::BaseMatrix::opMulTM(bresult,bm);
}

template<class Real>
template<typename real2>
void RotationMatrix<Real>::rotateSparseMatrix(
    linearalgebra::BaseMatrix * result,
    const SparseMatrix<real2>* Jmat)
{
    for (const auto& [rowId, row] : *Jmat)
    {
        const std::size_t nbIterationsOf3ConsecutiveElements = row.size() / 3;
        typename std::map<sofa::SignedIndex,real2>::const_iterator it = row.cbegin();

        for (std::size_t c = 0; c < nbIterationsOf3ConsecutiveElements; ++c)
        {
            const auto& [colId0, scalar0] = *it++;
            const auto& [colId1, scalar1] = *it++;
            const auto& [colId2, scalar2] = *it++;
            assert(colId1 == colId0 + 1);
            assert(colId2 == colId0 + 2);

            result->set(rowId,colId0,scalar0 * data[colId0*3+0] + scalar1 * data[colId1*3+0] + scalar2 * data[colId2*3+0] );
            result->set(rowId,colId1,scalar0 * data[colId0*3+1] + scalar1 * data[colId1*3+1] + scalar2 * data[colId2*3+1] );
            result->set(rowId,colId2,scalar0 * data[colId0*3+2] + scalar1 * data[colId1*3+2] + scalar2 * data[colId2*3+2] );
        }
    }
}

template<class Real>
void RotationMatrix<Real>::rotateMatrix(linearalgebra::BaseMatrix * mat,const linearalgebra::BaseMatrix * Jmat)
{
    if (mat != Jmat)
    {
        mat->clear();
        mat->resize(Jmat->rowSize(),Jmat->colSize());
    }

    if (const auto* Jf = dynamic_cast<const SparseMatrix<float> * >(Jmat))
    {
        rotateSparseMatrix(mat, Jf);
    }
    else if (const auto* Jd = dynamic_cast<const SparseMatrix<double> * >(Jmat))
    {
        rotateSparseMatrix(mat, Jd);
    }
    else
    {
        dmsg_warning("RotationMatrix") << "rotateMatrix for this kind of matrix is not implemented" ;
    }
}

template<class Real>
std::ostream& operator << (std::ostream& out, const RotationMatrix<Real> & v )
{
    out.precision(4);
    out << "[";
    for (unsigned y=0; y<v.data.size(); y+=9)
    {
        for (sofa::SignedIndex x=0; x<3; ++x)
        {
            out << "\n[" << std::fixed << v.data[y+x*3] << " " << std::fixed << v.data[y+x*3+1] << " " << std::fixed << v.data[y+x*3+2] << "]";
        }
    }
    out << "\n]";
    return out;
}

template<>
const char* RotationMatrix<float>::Name() {
    return "RotationMatrixf";
}

template<>
const char* RotationMatrix<double>::Name() {
    return "RotationMatrixd";
}


template class SOFA_LINEARALGEBRA_API RotationMatrix<float>;
template class SOFA_LINEARALGEBRA_API RotationMatrix<double>;

template SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const RotationMatrix<float>& v);
template SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const RotationMatrix<double>& v);


} // namespace sofa::component::solver
