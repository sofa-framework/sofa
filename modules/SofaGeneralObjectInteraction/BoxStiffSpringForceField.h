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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaDeformable/StiffSpringForceField.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/** Set springs between the particles located inside a given box.
*/
template <class DataTypes>
class BoxStiffSpringForceField : public sofa::component::interactionforcefield::StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BoxStiffSpringForceField, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    typedef StiffSpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename Inherit::Spring Spring;
    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef defaulttype::Vec<6,Real> Vec6;

protected:

    //float Xmin,Xmax,Ymin,Ymax,Zmin,Zmax;



    BoxStiffSpringForceField(MechanicalState* object1, MechanicalState* object2, double ks=100.0, double kd=5.0);
    BoxStiffSpringForceField(double ks=100.0, double kd=5.0);
public:
    void init() override;
    void bwdInit() override;

    Data<Vec6>  box_object1;
    Data<Vec6>  box_object2;
    Data<SReal> factorRestLength;
    Data<bool>  forceOldBehavior;
    // -- VisualModel interface

    void draw(const core::visual::VisualParams* vparams) override;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec2dTypes>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec1dTypes>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec2fTypes>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec1fTypes>;
extern template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<defaulttype::Vec6fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_H */
