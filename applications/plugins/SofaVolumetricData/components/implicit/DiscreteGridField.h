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

#ifndef SOFAVOLUMETRICDATA_IMPLICIT_DISCRETEGRIDFIELD_H
#define SOFAVOLUMETRICDATA_IMPLICIT_DISCRETEGRIDFIELD_H

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/ObjectFactory.h>
#include "ScalarField.h"
#include "../../DistanceGrid.h"

namespace sofa
{

namespace component
{

namespace implicit
{

using sofa::core::objectmodel::DataFileName ;
using sofa::component::container::DistanceGrid ;
using sofa::component::implicit::Vector3 ;

class DiscreteGridField : public ScalarField {
public:
    SOFA_CLASS(DiscreteGridField, ScalarField);

    DiscreteGridField();
    virtual ~DiscreteGridField()  { }
    DistanceGrid* grid {nullptr};
    void setFilename(const std::string& name);
    void loadGrid(double scale, double sampling, int nx, int ny, int nz, Vector3 pmin, Vector3 pmax);
    virtual void init();
    virtual double eval(Vector3 p);

private:
    DataFileName in_filename;
    Vector3 pmin, pmax;
    Data<int> in_nx;
    Data<int> in_ny;
    Data<int> in_nz;
    Data<double> in_scale;
    Data<double> in_sampling;

};

} /// implicit
    using implicit::DiscreteGridField ;

} /// component

} /// sofa

#endif
