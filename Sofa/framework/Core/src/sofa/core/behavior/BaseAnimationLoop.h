/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component responsible for main animation algorithms, managing how
 *  and when mechanical computation happens in one animation step.
 *
 *  This class can optionally replace the default computation scheme of computing
 *  collisions then doing an integration step.
 *
 *  Note that it is in a preliminary stage, hence its functionalities and API will
 *  certainly change soon.
 *
 */
class SOFA_CORE_API BaseAnimationLoop : public virtual objectmodel::BaseObject
{

public:
    SOFA_ABSTRACT_CLASS(BaseAnimationLoop, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseAnimationLoop)

    // the node where the loop will start processing.
    SingleLink<BaseAnimationLoop, core::objectmodel::BaseNode, BaseLink::FLAG_STOREPATH> l_node;

protected:
    BaseAnimationLoop();

    ~BaseAnimationLoop() override;

    /// Stores starting time of the simulation
    SReal m_resetTime;

    /// Save the initial state for later uses in reset()
    void storeResetState() override;

    virtual void doStep(const core::ExecParams* params, SReal dt) = 0 ;


private:
    BaseAnimationLoop(const BaseAnimationLoop& n) = delete ;
    BaseAnimationLoop& operator=(const BaseAnimationLoop& n) = delete ;

public:
    void init() override;

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doStep" internally,
     * which is the method to override from now on.
     *
     * Main computation method.
     *
     * Specify and execute all computations for computing a timestep, such
     * as one or more collisions and integrations stages.
     **/  
    virtual void step(const core::ExecParams* params, SReal dt) final {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doStep(params, dt);
    };

    /// Returns starting time of the simulation
    SReal getResetTime() const;

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

    Data<bool> d_computeBoundingBox; ///< If true, compute the global bounding box of the scene at each time step. Used mostly for rendering.

};

} // namespace sofa::core::behavior
