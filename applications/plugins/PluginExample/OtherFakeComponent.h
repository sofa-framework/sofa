/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_CONSTRAINT_OtherFakeComponent_H
#define SOFA_COMPONENT_CONSTRAINT_OtherFakeComponent_H

#include "initPluginExample.h"
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template <class DataTypes>
class  OtherFakeComponent : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OtherFakeComponent,DataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));
    typedef typename  DataTypes::VecDeriv VecDeriv;
    typedef typename  DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename  DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename  DataTypes::VecCoord VecCoord;
    OtherFakeComponent();
    ~OtherFakeComponent();

    void init();

    void reinit();

    void projectResponse(MatrixDerivRowType& /*dx*/) {}
    void projectResponse(VecDeriv& /*dx*/) {}
    void projectVelocity(VecDeriv& /*dx*/) {}
    void projectPosition(VecCoord& /*x*/) {}


protected:


private:

};

#if defined(WIN32) && !defined(SOFA_BUILD_PLUGINEXAMPLE)
#ifndef SOFA_FLOAT
extern template class OtherFakeComponent<defaulttype::Vec3dTypes>;
extern template class OtherFakeComponent<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class OtherFakeComponent<defaulttype::Vec3fTypes>;
extern template class OtherFakeComponent<defaulttype::Rigid3fTypes>;
#endif
#endif
}

}

}



#endif
