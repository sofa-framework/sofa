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
#include <sofa/core/VecId.h>

namespace sofa::core
{
/**
 *  \brief Component storing position and velocity vectors.
 *
 *  This class define the interface of components used as source and
 *  destination of regular (non mechanical) mapping. It is then specialized as
 *  MechanicalState (storing other mechanical data) or MappedModel (if no
 *  mechanical data is used, such as for VisualModel).
 */
class SOFA_CORE_API BaseState : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseState, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseState)
protected:
    BaseState() {}
    ~BaseState() override {}
    virtual void doResize(Size vsize) = 0;
    virtual objectmodel::BaseData* doBaseWrite(VecId v) = 0;
    virtual const objectmodel::BaseData* doBaseRead(ConstVecId v) const = 0;
    virtual void doAddToTotalForces(core::ConstVecDerivId forceId);
    virtual void doRemoveFromTotalForces(core::ConstVecDerivId forceId);
private:
    BaseState(const BaseState& n) = delete;
    BaseState& operator=(const BaseState& n) = delete;
public:
    /// Current size of all stored vectors
    virtual Size getSize() const = 0;

    /// Resize all stored vector
    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doResize" internally,
     * which is the method to override from now on.
     *
    **/
    virtual void resize(Size vsize) final
    {
        //TODO (SPRINT SED 2025): Component state mechanism
        doResize(vsize);
    }

    /// @name BaseData vectors access API based on VecId
    /// @{
    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doBaseWrite" internally,
     * which is the method to override from now on.
     *
    **/
    virtual objectmodel::BaseData* baseWrite(VecId v) final
    {
        //TODO (SPRINT SED 2025): Component state mechanism
        return doBaseWrite(v);
    }

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doBaseRead" internally,
     * which is the method to override from now on.
     *
    **/
    virtual const objectmodel::BaseData* baseRead(ConstVecId v) const final
    {
        //TODO (SPRINT SED 2025): Component state mechanism
        return doBaseRead(v);
    }

    /// @}


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

    /// The given VecDerivId is appended to a list representing all the forces containers
    /// It is useful to be able to compute the accumulation of all forces (for example the ones
    /// coming from force fields and the ones coming from lagrangian constraints).

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doAddToTotalForces" internally,
     * which is the method to override from now on.
     *
    **/
    virtual void addToTotalForces(core::ConstVecDerivId forceId) final
    {
        //TODO (SPRINT SED 2025): Component state mechanism
        doAddToTotalForces(forceId);
    }
    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doRemoveFromTotalForces" internally,
     * which is the method to override from now on.
     *
    **/
    virtual void removeFromTotalForces(core::ConstVecDerivId forceId) final
    {
        //TODO (SPRINT SED 2025): Component state mechanism
        doRemoveFromTotalForces(forceId);
    }
};

} // namespace sofa::core

