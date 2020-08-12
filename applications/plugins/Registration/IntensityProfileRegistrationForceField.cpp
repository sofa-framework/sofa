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

      // Register in the Factory
      int IntensityProfileRegistrationForceFieldClass = core::RegisterObject("Compute normal forces on a point set based on the closest intensity profile in the target image")
          .add< IntensityProfileRegistrationForceField<Vec3Types,ImageUC> >(true)
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageUS> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageD> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageC> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageI> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageUI> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageS> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageL> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageUL> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageF> >()
      .add< IntensityProfileRegistrationForceField<Vec3Types,ImageB> >()

    
    ;

          template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageUC>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageUS>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageD>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageC>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageI>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageUI>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageS>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageL>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageUL>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageF>;
      template class SOFA_REGISTRATION_API IntensityProfileRegistrationForceField<Vec3Types,ImageB>;

    


            
}
}

} // namespace sofa


