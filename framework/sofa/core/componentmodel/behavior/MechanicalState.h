/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MECHANICALSTATE_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MECHANICALSTATE_H

#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class TDataTypes>
class MechanicalState : public BaseMechanicalState
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::SparseDeriv SparseDeriv;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::VecConst VecConst;

    virtual ~MechanicalState() { }

    virtual VecCoord* getX() = 0;
    virtual VecDeriv* getV() = 0;
    virtual VecDeriv* getF() = 0;
    virtual VecDeriv* getDx() = 0;
    virtual VecConst* getC() = 0;
    virtual VecCoord* getXfree() = 0;

    virtual const VecCoord* getX()  const = 0;
    virtual const VecDeriv* getV()  const = 0;
    virtual const VecDeriv* getF()  const = 0;
    virtual const VecDeriv* getDx() const = 0;
    virtual const VecConst* getC() const = 0;
    virtual const VecCoord* getXfree()  const = 0;

    virtual const VecCoord* getX0()  const = 0;
    virtual const VecDeriv* getV0()  const = 0;


    /// Get the indices of the particles located in the given bounding box
    virtual void getIndicesInSpace(std::vector<unsigned>& /*indices*/, Real /*xmin*/, Real /*xmax*/,Real /*ymin*/, Real /*ymax*/, Real /*zmin*/, Real /*zmax*/) const=0;

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MechanicalState<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
