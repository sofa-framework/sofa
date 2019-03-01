/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H
#include "config.h"

#include <sofa/core/collision/ContactManager.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/map_ptr_stable_compare.h>


namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_BASE_COLLISION_API DefaultContactManager : public core::collision::ContactManager
{
public :
    SOFA_CLASS(DefaultContactManager,sofa::core::collision::ContactManager);

    Data<sofa::helper::OptionsGroup> response; ///< contact response class
    Data<std::string> responseParams; ///< contact response parameters (syntax: name1=value1    Data<std::string> responseParams;name2=value2    Data<std::string> responseParams;...)

    /// outputsVec fixes the reproducibility problems by storing contacts in the collision detection saved order
    /// if not given, it is still working but with eventual reproducibility problems
    void createContacts(const DetectionOutputMap& outputs) override;
    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;

    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        if (context)
        {
            context->addObject(obj);
            core::collision::Pipeline *pipeline = static_cast<simulation::Node*>(context)->collisionPipeline;
            sofa::helper::OptionsGroup options = initializeResponseOptions(pipeline);
            obj->response.setValue(options);
        }

        if (arg)
            obj->parse(arg);

        return obj;
    }

    void reset() override;
    void cleanup() override;

    virtual std::string getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2) override;

    /// virtual methods used for cleaning the pipeline after a dynamic graph node deletion.
    /**
     * Contacts can be attached to a deleted node and their deletion is a problem for the pipeline.
     * @param c is the list of deleted contacts.
     */
    void removeContacts(const ContactVector &/*c*/) override;

    void setDefaultResponseType(const std::string &responseT);

    std::string getDefaultResponseType() const { return response.getValue().getSelectedItem(); }

protected:
    typedef sofa::helper::map_ptr_stable_compare<std::pair<core::CollisionModel*,
    core::CollisionModel*>,core::collision::Contact::SPtr> ContactMap;

    DefaultContactManager();
    ~DefaultContactManager() override;

    void setContactTags(core::CollisionModel* model1, core::CollisionModel* model2,
                        core::collision::Contact::SPtr contact);

    ContactMap contactMap;
    std::map<Instance,ContactMap> storedContactMap;

    void changeInstance(Instance inst) override ;

    static sofa::helper::OptionsGroup initializeResponseOptions(core::collision::Pipeline *pipeline);

    // count failure messages, so we don't continuously repeat them
    std::map<std::pair<std::string,std::pair<std::string,std::string> >, int> errorMsgCount;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
