/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_ROTATIONMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_ROTATIONMATRIX_H

#include <sofa/component/linearsolver/SparseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library
template<class Real>
class RotationMatrix : public defaulttype::BaseMatrix
{
public:

    virtual unsigned int rowSize(void) const
    {
        return data.size()/3;
    }

    /// Number of columns
    virtual unsigned int colSize(void) const
    {
        return data.size()/3;
    }

    /// Read the value of the element at row i, column j (using 0-based indices)
    virtual SReal element(int i, int j) const
    {
        int bd = (i/3)*3;
        if ((j<bd) || (j>bd+2)) return 0.0 ;

        return data[i*3+j-bd];
    }

    /// Resize the matrix and reset all values to 0
    virtual void resize(int nbRow, int nbCol)
    {
        if (nbRow!=nbCol) return;
        data.resize(nbRow*3);
    }

    /// Reset all values to 0
    virtual void clear()
    {
        data.clear();
    }

    /// Write the value of the element at row i, column j (using 0-based indices)
    virtual void set(int i, int j, double v)
    {
        int bd = (i/3)*3;
        if ((j<bd) || (j>bd+2)) return;
        data[i*3+j-bd] = (Real)v;
    }

    /// Add v to the existing value of the element at row i, column j (using 0-based indices)
    virtual void add(int i, int j, double v)
    {
        int bd = (i/3)*3;
        if ((j<bd) || (j>bd+2)) return;

        data[i*3+j-bd] += (Real)v;
    }

    virtual helper::vector<Real> & getVector()
    {
        return data;
    }

    virtual void opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const
    {
        //Solve lv = R * lvR
        unsigned int k = 0,l = 0;
        while (k < data.size())
        {
            result->set(l+0,data[k + 0] * v->element(l+0) + data[k + 1] * v->element(l+1) + data[k + 2] * v->element(l+2));
            result->set(l+1,data[k + 3] * v->element(l+0) + data[k + 4] * v->element(l+1) + data[k + 5] * v->element(l+2));
            result->set(l+2,data[k + 6] * v->element(l+0) + data[k + 7] * v->element(l+1) + data[k + 8] * v->element(l+2));
            l+=3;
            k+=9;
        }
    }

    virtual void opMulTV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const
    {
        unsigned int k = 0,l = 0;
        while (k < data.size())
        {
            result->set(l+0,data[k + 0] * v->element(l+0) + data[k + 3] * v->element(l+1) + data[k + 6] * v->element(l+2));
            result->set(l+1,data[k + 1] * v->element(l+0) + data[k + 4] * v->element(l+1) + data[k + 7] * v->element(l+2));
            result->set(l+2,data[k + 2] * v->element(l+0) + data[k + 5] * v->element(l+1) + data[k + 8] * v->element(l+2));
            l+=3;
            k+=9;
        }
    }

    /// multiply the transpose current matrix by m matrix and strore the result in m
    virtual void opMulTM(defaulttype::BaseMatrix * bresult,defaulttype::BaseMatrix * bm)
    {
        if (RotationMatrix<Real> * m = dynamic_cast<RotationMatrix<Real> * >(bm))
        {
            if (RotationMatrix<Real> * result = dynamic_cast<RotationMatrix<Real> * >(bresult))
            {
                Real tmp[9];
                for (unsigned i=0; i<data.size(); i+=9)
                {
                    tmp[0] = data[i+0] * m->data[i+0] + data[i+3] * m->data[i+3] + data[i+6] * m->data[i+6];
                    tmp[1] = data[i+0] * m->data[i+1] + data[i+3] * m->data[i+4] + data[i+6] * m->data[i+7];
                    tmp[2] = data[i+0] * m->data[i+2] + data[i+3] * m->data[i+5] + data[i+6] * m->data[i+8];

                    tmp[3] = data[i+1] * m->data[i+0] + data[i+4] * m->data[i+3] + data[i+7] * m->data[i+6];
                    tmp[4] = data[i+1] * m->data[i+1] + data[i+4] * m->data[i+4] + data[i+7] * m->data[i+7];
                    tmp[5] = data[i+1] * m->data[i+2] + data[i+4] * m->data[i+5] + data[i+7] * m->data[i+8];

                    tmp[6] = data[i+2] * m->data[i+0] + data[i+5] * m->data[i+3] + data[i+8] * m->data[i+6];
                    tmp[7] = data[i+2] * m->data[i+1] + data[i+5] * m->data[i+4] + data[i+8] * m->data[i+7];
                    tmp[8] = data[i+2] * m->data[i+2] + data[i+5] * m->data[i+5] + data[i+8] * m->data[i+8];

                    result->data[i+0] = tmp[0]; result->data[i+1] = tmp[1]; result->data[i+2] = tmp[2];
                    result->data[i+3] = tmp[3]; result->data[i+4] = tmp[4]; result->data[i+5] = tmp[5];
                    result->data[i+6] = tmp[6]; result->data[i+7] = tmp[7]; result->data[i+8] = tmp[8];
                }
                return;
            }
        }
        defaulttype::BaseMatrix::opMulTM(bresult,bm);
    }

    template<typename TReal, typename JMatrix>
    void opMulJ(typename component::linearsolver::SparseMatrix<TReal> * JR,JMatrix * J)
    {
        JR->clear();
        JR->resize(J->rowSize(),J->colSize());

        //compute JR = J * R
        for (typename JMatrix::LineConstIterator jit1 = J->begin(); jit1 != J->end(); jit1++)
        {
            int l = jit1->first;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                int c = i1->first;
                Real v0 = (Real)i1->second; i1++; if (i1==jit1->second.end()) break;
                Real v1 = (Real)i1->second; i1++; if (i1==jit1->second.end()) break;
                Real v2 = (Real)i1->second; i1++;
                JR->set(l,c+0,v0 * data[(c+0)*3+0] + v1 * data[(c+1)*3+0] + v2 * data[(c+2)*3+0] );
                JR->set(l,c+1,v0 * data[(c+0)*3+1] + v1 * data[(c+1)*3+1] + v2 * data[(c+2)*3+1] );
                JR->set(l,c+2,v0 * data[(c+0)*3+2] + v1 * data[(c+1)*3+2] + v2 * data[(c+2)*3+2] );
            }
        }
    }

    friend std::ostream& operator << (std::ostream& out, const RotationMatrix<Real> & v )
    {
        out << "[";
        for (unsigned y=0; y<v.data.size(); y+=9)
        {
            for (int x=0; x<3; ++x)
            {
                out << "\n[" << v.data[y+x*3] << " " << v.data[y+x*3+1] << " " << v.data[y+x*3+2] << "]";
            }
        }
        out << "\n]";
        return out;
    }

private :
    helper::vector<Real> data;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
