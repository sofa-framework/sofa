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
#ifndef SOFA_CORE_FEM_BASEMATERIAL_H
#define SOFA_CORE_FEM_BASEMATERIAL_H

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>


namespace sofa
{
namespace component
{
namespace fem
{
namespace material
{


using namespace sofa::defaulttype;
/**
 * Generic material class
 */
class SOFA_CORE_API BaseMaterial : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseMaterial,core::objectmodel::BaseObject);

    BaseMaterial();
    virtual ~BaseMaterial() {}

    virtual void init();


    //virtual VecN computeStress (VecN & strain,int idElement,int id_QP){return stress in the i-th quadrature point}
    //So here needed the shapefunctionvalue *  ,  quadratureformular*  (verifie if shapfunctionvalue compute with the local method)
    // The same principe for computing the strain given the displacement


    virtual void computeStress (Vector3 & ,Vector3 &,unsigned int &) {}
    virtual void computeDStress (Vector3 & ,Vector3 &) {}

    virtual void computeStress (unsigned int /*iElement*/)=0;//to be pure virtual
    virtual void handleTopologyChange();
};

}

} // namespace fem

} // namespace core

} // namespace sofa
#endif
