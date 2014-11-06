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
#ifndef SOFA_COMPONENT_MAPPEDOBJECT_H
#define SOFA_COMPONENT_MAPPEDOBJECT_H

#include <sofa/core/State.h>
#include <sofa/SofaBase.h>
#include <vector>
#include <assert.h>
#include <fstream>

namespace sofa
{

namespace component
{

namespace container
{
using core::objectmodel::Data;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class MappedObjectInternalData
{
public:
};

template <class DataTypes>
class MappedObject : public core::State<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MappedObject,DataTypes), SOFA_TEMPLATE(core::State,DataTypes));
    typedef core::State<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:

    MappedObjectInternalData<DataTypes> data;



    MappedObject();

    virtual ~MappedObject();
public:
    virtual void init();

    Data<VecCoord> f_X;
    Data<VecDeriv> f_V;

    virtual void resize(int vsize) { f_X.beginEdit()->resize(vsize); f_X.endEdit(); f_V.beginEdit()->resize(vsize); f_V.endEdit(); }

    VecCoord* getX()  { return f_X.beginEdit(); }
    VecDeriv* getV()  { return f_V.beginEdit(); }

    const VecCoord* getX()  const { return &f_X.getValue();  }
    const VecDeriv* getV()  const { return &f_V.getValue();  }

    int getSize() const
    {
        return f_X.getValue().size();
    }

    Data< VecCoord >* write(core::VecCoordId v)
    {
        if(v == core::VecCoordId::position())
            return &f_X;

        return NULL;
    }

    const Data< VecCoord >* read(core::ConstVecCoordId v) const
    {
        if(v == core::ConstVecCoordId::position())
            return &f_X;
        else
            return NULL;
    }

    Data< VecDeriv >* write(core::VecDerivId v)
    {
        if(v == core::VecDerivId::velocity())
            return &f_V;
        else
            return NULL;
    }

    const Data< VecDeriv >* read(core::ConstVecDerivId v) const
    {
        if(v == core::ConstVecDerivId::velocity())
            return &f_V;
        else
            return NULL;
    }

    Data< MatrixDeriv >* write(core::MatrixDerivId /*v*/)
    {
        return NULL;
    }

    const Data< MatrixDeriv >* read(core::ConstMatrixDerivId /*v*/) const
    {
        return NULL;
    }
};

}

} // namespace component

} // namespace sofa

#endif
