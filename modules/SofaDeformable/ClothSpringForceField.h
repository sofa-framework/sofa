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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHSPRINGFORCEFIELD_H

#include <SofaDeformable/StiffSpringForceField.h>
#include <set>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/***
This force field add linear spring which forbids only elongation on each point to connect them with all their neighbors belonging the the same faces, 
even if there is no topology edges linking them. On a regular grid each points are connected to eight others.
This network is usefull for cloth simulation, it correspond to stretch spring described in the paper of Choi and al 
"Stable but Responsive Cloth" (http://graphics.snu.ac.kr/~kjchoi/publication/cloth.pdf)         
*/

template<class DataTypes>
class ClothSpringForceField : public sofa::component::interactionforcefield::StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ClothSpringForceField, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

public:
    /// create the spring network 
    virtual void init();

protected:
    ClothSpringForceField(double ks=100.0, double kd=5.0): 
        StiffSpringForceField<DataTypes>(ks, kd)
    {

    }

   
    ~ClothSpringForceField()
    {

    }

    typedef std::pair<unsigned,unsigned> IndexPair;
    void addSpring( unsigned, unsigned, std::set<IndexPair>& );

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHSPRINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_DEFORMABLE_API ClothSpringForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_DEFORMABLE_API ClothSpringForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_QUADBENDINGSPRINGS_H */
