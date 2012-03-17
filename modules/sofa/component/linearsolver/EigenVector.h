/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenVector_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenVector_H

#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
#include <Eigen/Dense>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define EigenVector_CHECK
//#define EigenVector_VERBOSE


/** Container of an Eigen vector.
  */
template<class InDataTypes>
class EigenVector : public defaulttype::BaseVector
{

protected:
    typedef typename InDataTypes::Real Real;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    VectorEigen eigenVector;    ///< the data


public:
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size };
    typedef defaulttype::Vec<Nin,Real> Block;

    VectorEigen& getVectorEigen() { return eigenVector; }
    const VectorEigen& getVectorEigen() const { return eigenVector; }



    EigenVector(int nbRow=0)
    {
        resize(nbRow);
    }

    unsigned size() const { return eigenVector.size(); }

    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(int nbRow)
    {
        eigenVector.resize(nbRow);
    }

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlocks)
    {
        eigenVector.resize(nbBlocks * Nin);
    }




    SReal element(int i) const
    {
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        return eigenVector.coeff(i);
    }

    void set(int i, double v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) = (Real)v;
    }

    void setBlock(int i, const Block& v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize()/Nout || (unsigned)j >= (unsigned)colSize()/Nin )
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()/Nout<<","<<colSize()/Nin<<")"<<std::endl;
            return;
        }
#endif
        for(unsigned l=0; l<Nin; l++)
            eigenVector.coeffRef(Nin*i+l) = (Real) v[l];
    }




    void add(int i, double v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() << */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "/*<<this->Name()*/<<" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) += (Real)v;
    }

    void clear(int i)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0"<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) = (Real)0;
    }


    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear()
    {
        resize(0,0);
        resize(size());
    }


    friend std::ostream& operator << (std::ostream& out, const EigenVector<InDataTypes>& v )
    {
        int ny = v.size();
        for (int y=0; y<ny; ++y)
        {
            out << " " << v.element(y);
        }
        return out;
    }

    static const char* Name();


};

template<> inline const char* EigenVector<defaulttype::Vec3dTypes>::Name() { return "EigenVector3d"; }
template<> inline const char* EigenVector<defaulttype::Vec3fTypes>::Name() { return "EigenVector3f"; }







/** Container of an Eigen vector.
  */
template<>
class EigenVector<double> : public defaulttype::BaseVector
{

protected:
    typedef double Real;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    VectorEigen eigenVector;    ///< the data


public:

    VectorEigen& getVectorEigen() { return eigenVector; }
    const VectorEigen& getVectorEigen() const { return eigenVector; }


    unsigned size() const { return eigenVector.size(); }

    EigenVector(int nbRow=0)
    {
        resize(nbRow);
    }

    /// Resize the matrix without preserving the data
    void resize(int nbRow)
    {
        eigenVector.resize(nbRow);
    }



    SReal element(int i) const
    {
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        return eigenVector.coeff(i);
    }

    void set(int i, double v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) = (Real)v;
    }





    void add(int i, double v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() << */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "/*<<this->Name()*/<<" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) += (Real)v;
    }

    void clear(int i)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0"<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenVector.coeffRef(i) = (Real)0;
    }


    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear()
    {
        int currentSize = size();
        resize(0);
        resize( currentSize );
    }


    friend std::ostream& operator << (std::ostream& out, const EigenVector<double>& v )
    {
        int ny = v.size();
        for (int y=0; y<ny; ++y)
        {
            out << " " << v.element(y);
        }
        return out;
    }

    static const char* Name();


};

const char* EigenVector<double>::Name() { return "EigenVectord"; }

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
