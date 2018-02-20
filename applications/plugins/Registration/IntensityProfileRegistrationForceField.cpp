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
#define INTENSITYPROFILEREGISTRATIONFORCEFIELD_CPP

#include "IntensityProfileRegistrationForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

    using namespace defaulttype;

      SOFA_DECL_CLASS(IntensityProfileRegistrationForceField)

      // Register in the Factory
      int IntensityProfileRegistrationForceFieldClass = core::RegisterObject("Compute normal forces on a point set based on the closest intensity profile in the target image")
    #ifndef SOFA_FLOAT
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageUC> >(true)
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageUS> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageD> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageC> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageI> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageUI> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageS> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageL> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageUL> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageF> >()
      .add< IntensityProfileRegistrationForceField<Vec3dTypes,ImageB> >()

    #endif
    #ifndef SOFA_DOUBLE
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageUC> >(true)
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageUS> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageD> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageC> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageI> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageUI> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageS> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageL> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageUL> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageF> >()
      .add< IntensityProfileRegistrationForceField<Vec3fTypes,ImageB> >()

    #endif
    ;

    #ifndef SOFA_FLOAT
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageUC>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageUS>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageD>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageC>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageI>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageUI>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageS>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageL>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageUL>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageF>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3dTypes,ImageB>;

    #endif
    #ifndef SOFA_DOUBLE
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageUC>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageUS>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageD>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageC>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageI>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageUI>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageS>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageL>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageUL>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageF>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3fTypes,ImageB>;
#endif


            
}
}

} // namespace sofa


