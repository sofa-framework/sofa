/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenVector_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenVector_H

#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
#include <Eigen/Dense>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define EigenVector_CHECK


/** Container of a vector of the Eigen library. Not an eigenvector of a matrix.
  */
template<class InDataTypes>
class EigenVector : public defaulttype::BaseVector
{

protected:
    typedef typename InDataTypes::Real Real;
public:
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    typedef typename VectorEigen::Index  IndexEigen;
protected:
    VectorEigen eigenVector;    ///< the data


public:
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size };
    typedef defaulttype::Vec<Nin,Real> Block;

    VectorEigen& getVectorEigen() { return eigenVector; }
    const VectorEigen& getVectorEigen() const { return eigenVector; }



    EigenVector(Index nbRow=0)
    {
        resize(nbRow);
    }

    Index size() const { return eigenVector.size(); }

    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(Index nbRow)
    {
        eigenVector.resize((IndexEigen)nbRow);
    }

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(Index nbBlocks)
    {
        eigenVector.resize((IndexEigen)nbBlocks * Nin);
    }




    SReal element(Index i) const
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        return eigenVector.coeff((IndexEigen)i);
    }

    void set(Index i, double v)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) = (Real)v;
    }

    void setBlock(Index i, const Block& v)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize()/Nout || j >= colSize()/Nin )
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()/Nout<<","<<colSize()/Nin<<")"<<std::endl;
            return;
        }
#endif
        for(Index l=0; l<Nin; l++)
            eigenVector.coeffRef((IndexEigen)Nin*i+l) = (Real) v[l];
    }




    void add(Index i, double v)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "/*<<this->Name()*/<<" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) += (Real)v;
    }

    void clear(Index i)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) = (Real)0;
    }


    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear()
    {
        resize(0);
        resize(size());
    }


    friend std::ostream& operator << (std::ostream& out, const EigenVector<InDataTypes>& v )
    {
        IndexEigen ny = v.size();
        for (IndexEigen y=0; y<ny; ++y)
        {
            out << " " << v.element(y);
        }
        return out;
    }

    static const char* Name();


};

#ifndef SOFA_FLOAT
template<> const char* EigenVector<defaulttype::Vec3dTypes>::Name();
#endif
#ifndef SOFA_DOUBLE
template<> const char* EigenVector<defaulttype::Vec3fTypes>::Name();
#endif




/** Container of an Eigen vector.
  */
template<>
class EigenVector<double> : public defaulttype::BaseVector
{

protected:
    typedef double Real;

public:
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;

protected:
    VectorEigen eigenVector;    ///< the data


public:

    VectorEigen& getVectorEigen() { return eigenVector; }
    const VectorEigen& getVectorEigen() const { return eigenVector; }


    Index size() const { return eigenVector.size(); }

    EigenVector(Index nbRow=0)
    {
        resize(nbRow);
    }

    /// Resize the matrix without preserving the data
    void resize(Index nbRow)
    {
        eigenVector.resize(nbRow);
    }



    SReal element(Index i) const
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        return eigenVector.coeff(i);
    }

    void set(Index i, double v)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) = (Real)v;
    }





    void add(Index i, double v)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "/*<<this->Name()*/<<" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) += (Real)v;
    }

    void clear(Index i)
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) = (Real)0;
    }


    /// Set all values to 0
    void clear()
    {
        eigenVector.setZero();
    }

    static const char* Name();


};


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
