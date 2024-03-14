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
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/type/Vec.h>
#include <Eigen/Dense>

namespace sofa::linearalgebra
{



/** Container of a vector of the Eigen library. Not an eigenvector of a matrix.
  */
template<class TReal, std::size_t TBlockSize = 1>
class SOFA_LINEARALGEBRA_API EigenVector : public linearalgebra::BaseVector
{

protected:
    typedef TReal Real;
public:
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    typedef typename VectorEigen::Index  IndexEigen;
protected:
    VectorEigen eigenVector;    ///< the data


public:
    enum { Nin= TBlockSize };
    typedef type::Vec<Nin,Real> Block;

    VectorEigen& getVectorEigen() { return eigenVector; }
    const VectorEigen& getVectorEigen() const { return eigenVector; }

    EigenVector(Index nbRow=0)
    {
        resize(nbRow);
    }

    Index size() const override { return Index(eigenVector.size()); }

    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(Index nbRow) override
    {
        eigenVector.resize((IndexEigen)nbRow);
    }

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(Index nbBlocks)
    {
        eigenVector.resize((IndexEigen)nbBlocks * Nin);
    }




    SReal element(Index i) const override
    {
#if EIGEN_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVector") << "Invalid read access to element (" << i << "," << j << ") in " <</*this->Name()<<*/" of size (" << rowSize() << "," << colSize() << ")";
            return 0.0;
        }
#endif
        return eigenVector.coeff((IndexEigen)i);
    }

    void set(Index i, SReal v) override
    {
#if EIGEN_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVector") << "Invalid write access to element (" << i << "," << j << ") in " <</*this->Name()<<*/" of size (" << rowSize() << "," << colSize() << ")";
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) = (Real)v;
    }

    void setBlock(Index i, const Block& v)
    {
#if EIGEN_CHECK
        if (i >= rowSize()/Nout || j >= colSize()/Nin )
        {
            msg_error("EigenVector") << "Invalid write access to element (" << i << "," << j << ") in " <</*this->Name()<<*/" of size (" << rowSize() / Nout << "," << colSize() / Nin << ")";
            return;
        }
#endif
        for(Index l=0; l<Nin; l++)
            eigenVector.coeffRef((IndexEigen)Nin*i+l) = (Real) v[l];
    }




    void add(Index i, SReal v) override
    {
#if EIGEN_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVector") << "Invalid write access to element (" << i << "," << j << ") in "/*<<this->Name()*/ << " of size (" << rowSize() << "," << colSize() << ")";
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) += (Real)v;
    }

    void clear(Index i) override
    {
#if EIGEN_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVector") << "Invalid write access to element (" << i << "," << j << ") in " <</*this->Name()<<*/" of size (" << rowSize() << "," << colSize() << ")";
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) = (Real)0;
    }


    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear() override
    {
        resize(0);
        resize(size());
    }


    friend std::ostream& operator << (std::ostream& out, const EigenVector<TReal, TBlockSize>& v )
    {
        IndexEigen ny = v.size();
        for (IndexEigen y=0; y<ny; ++y)
        {
            out << " " << v.element(y);
        }
        return out;
    }

    static const std::string Name()
    {
        std::ostringstream o;
        o << "EigenVector";

        if constexpr (std::is_scalar<TReal>::value)
        {
            if constexpr (std::is_same<float, TReal>::value)
            {
                o << "f";
            }
            if constexpr (std::is_same<double, TReal>::value)
            {
                o << "d";
            }
        }

        return o.str();
    }


};

} // namespace sofa::linearalgebra
