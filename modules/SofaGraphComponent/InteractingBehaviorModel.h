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
#ifndef SOFA_CORE_INTERACTINGBEHAVIORMODEL_H
#define SOFA_CORE_INTERACTINGBEHAVIORMODEL_H
#include "config.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/behavior/Mass.h>


namespace sofa
{

namespace component
{

namespace misc
{



/**
 *  \brief Abstract Interface of components defining the behavior of a simulated object.
 *
 *  This Interface is used by "black-box" objects
 *  that are present in a SOFA simulation, which do not use the internal
 *  behavior components (MechanicalState, ForceField, etc), but which can
 *  interact with other objects.
 *
 *  It permits to include simulations from others libraries in a sofa scene.
 *
 *  On a SOFA point of view, the external behavior model is seen as a Mass (that is itself a ForceField)
 *  For the complete API (all ways to assemble the system, etc.), have a look to
 *    sofa/core/behavior/BaseMass.h
 *    sofa/core/behavior/Mass.h
 *    sofa/core/behavior/BaseForceField.h
 *    sofa/core/behavior/ForceField.h
 *
 *  @warning It is a priliminary version, the api must be improved to be able to handle contact dofs whith varying numbers and location
 *  and to handle visual & collision models
 *
 *  @param DataTypes template type encodes the dof type
 */
template<class DataTypes>
class InteractingBehaviorModel : public core::behavior::Mass<DataTypes>
{

public:

    // SOFA black magic
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(InteractingBehaviorModel, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;

    typedef component::container::MechanicalObject<DataTypes> Dofs;


    /// call when initializing the simulation
    virtual void init()
    {
        Inherited::init();
        m_exposedDofs = dynamic_cast<Dofs*>( this->getMState() );
    }


protected:

    InteractingBehaviorModel() : Inherited(), m_exposedDofs(0)
    {
        this->f_listening.setValue(true); // to call handleEvent at each event
    }

    virtual ~InteractingBehaviorModel() {}


    /// get an access to the exposed sofa dofs (useful to get object translation, rotation, rest positions...)
    Dofs* m_exposedDofs;


private:

    // no copy constructor
    InteractingBehaviorModel( const InteractingBehaviorModel& ) {}

};


} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_CORE_INTERACTINGBEHAVIORMODEL_H
