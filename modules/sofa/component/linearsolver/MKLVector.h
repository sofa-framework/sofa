/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MKLVECTOR_H
#define SOFA_COMPONENT_LINEARSOLVER_MKLVECTOR_H

#include <sofa/defaulttype/BaseVector.h>

#include <MKL/vect_dyn.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

class MKLVector : public defaulttype::BaseVector
{
    friend class MKLMatrix;

public:

    MKLVector()
    {
        impl = new Dynamic_Vector<double>();
    }

    virtual ~MKLVector()
    {
        delete impl;
    }

    virtual void resize(Index dim)
    {
        impl->resize(dim);
    };

    virtual double &element(Index i)
    {
        return impl->operator[](i);
    };

    virtual unsigned int size(void)
    {
        return impl->size();
    };

private:
    Dynamic_Vector<double> *impl;
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
