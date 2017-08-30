/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
/******************************************************************************
* Contributors:                                                               *
*   - thomas.goss@etudiant.univ-lille1.fr                                     *
*   - damien.marchal@univ-lille1.fr                                           *
******************************************************************************/
#include "DiscreteGridField.h"

#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::ComponentState ;

namespace sofa
{

namespace component
{

namespace implicit
{

DiscreteGridField::DiscreteGridField()
    : in_filename(initData(&in_filename,"filename","filename"))
    , in_nx(initData(&in_nx,0,"nx","in_nx"))
    , in_ny(initData(&in_ny,0,"ny","in_ny"))
    , in_nz(initData(&in_nz,0,"nz","in_nz"))
    , in_scale(initData(&in_scale,0.0,"scale","in_scale"))
    , in_sampling(initData(&in_sampling,0.0,"sampling","in_sampling"))
{
}


///fil the grid
void DiscreteGridField::loadGrid(double scale, double sampling, int nx, int ny, int nz, Vector3 pmin, Vector3 pmax)
{
    grid = grid->load(in_filename.getFullPath(), scale, sampling, nx, ny, nz, pmin, pmax);
    if(grid == nullptr) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "uninitialized grid";
    }

    return;
}


///used to set a name in tests
void DiscreteGridField::setFilename(const std::string& name)
{
    this->in_filename.setValue(name);
}


///eval the grid
double DiscreteGridField::eval(Vector3 p)
{
    if(m_componentstate != ComponentState::Valid)
    {
        msg_error() << "Trying to evalute a non valid distance grid always returns 0.0" ;
        return 0.0;
    }

    return grid->quickeval(p);
}


void DiscreteGridField::init()
{
    if(in_nx.getValue()==0 && in_nz.getValue()==0 && in_nz.getValue()==0) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "uninitialized grid";
    }
    else if(in_filename.isSet() == false) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "unset filename";
    }
    else {
        pmin.set(0,0,-5.0);
        pmax.set(27,27,5.0);
        loadGrid(in_scale.getValue(),in_sampling.getValue(),in_nx.getValue(),in_ny.getValue(),in_nz.getValue(),pmin,pmax);
    }

    m_componentstate = ComponentState::Valid;
}

///factory register
int DistanceGridComponentClass = RegisterObject("A discrete scalar field from a regular grid storing field value with interpolation.").add< DiscreteGridField >().addAlias("DistGrid");


} ///namespace implicit

} ///namespace core

} ///namespace sofa
