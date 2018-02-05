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
#ifndef SOFA_CORE_BEHAVIOR_BASEANIMATIONLOOP_H
#define SOFA_CORE_BEHAVIOR_BASEANIMATIONLOOP_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ExecParams.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component responsible for main animation algorithms, managing how
 *  and when mechanical computation happens in one animation step.
 *
 *  This class can optionally replace the default computation scheme of computing
 *  collisions then doing an integration step.
 *
 *  Note that it is in a preliminary stage, hence its fonctionnalities and API will
 *  certainly change soon.
 *
 */

class SOFA_CORE_API BaseAnimationLoop : public virtual objectmodel::BaseObject
{

public:
    SOFA_ABSTRACT_CLASS(BaseAnimationLoop, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseAnimationLoop)

protected:
    BaseAnimationLoop();

    virtual ~BaseAnimationLoop();

    /// Stores starting time of the simulation
    SReal m_resetTime;

    /// Save the initial state for later uses in reset()
    virtual void storeResetState() override;
	
	
private:
	BaseAnimationLoop(const BaseAnimationLoop& n) ;
	BaseAnimationLoop& operator=(const BaseAnimationLoop& n) ;

public:
    /// Main computation method.
    ///
    /// Specify and execute all computations for computing a timestep, such
    /// as one or more collisions and integrations stages.
    virtual void step(const core::ExecParams* params, SReal dt) = 0;

    /// Returns starting time of the simulation
    SReal getResetTime() const;

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif /* SOFA_CORE_BEHAVIOR_BASEANIMATIONLOOP_H */
