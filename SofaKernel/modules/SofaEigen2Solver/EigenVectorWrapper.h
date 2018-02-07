/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenVectorWrapper_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenVectorWrapper_H

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


/** Wrapper of an Eigen vector to provide it with a defaulttype::BaseVector interface.
  */
template<class Real>
class EigenVectorWrapper : public defaulttype::BaseVector
{

public:
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    typedef typename VectorEigen::Index  IndexEigen;
protected:
    VectorEigen& eigenVector;    ///< the data


public:

    VectorEigen& getVectorEigen() { return eigenVector; }
    const VectorEigen& getVectorEigen() const { return eigenVector; }



    EigenVectorWrapper( VectorEigen& ve ): eigenVector(ve) {}

    Index size() const { return eigenVector.size(); }

    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(Index nbRow)
    {
        eigenVector.resize((IndexEigen)nbRow);
    }



    SReal element(Index i) const
    {
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVectorWrapper") << "Invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")" ;
            return 0.0;
        }
#endif
        return eigenVector.coeff((IndexEigen)i);
    }

    void set(Index i, double v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVectorWrapper") << "Invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")" ;
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) = (Real)v;
    }





    void add(Index i, double v)
    {
#ifdef EigenVector_VERBOSE
        std::cout << /*this->Name() << */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v<<std::endl;
#endif
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVectorWrapper") << "Invalid write access to element ("<<i<<","<<j<<") in "/*<<this->Name()*/<<" of size ("<<rowSize()<<","<<colSize()<<")" ;
            return;
        }
#endif
        eigenVector.coeffRef((IndexEigen)i) += (Real)v;
    }

    void clear(Index i)
    {
#ifdef EigenVector_VERBOSE
        msg_error("EigenVectorWrapper") << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0" ;
#endif
#ifdef EigenVector_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            msg_error("EigenVectorWrapper") << "Invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")" ;
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


    friend std::ostream& operator << (std::ostream& out, const EigenVectorWrapper<Real>& v )
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

//  template<> const char* EigenVectorWrapper<defaulttype::float>::Name();
//  template<> const char* EigenVectorWrapper<defaulttype::double>::Name();



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
