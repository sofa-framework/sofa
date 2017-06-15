/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H
#include "config.h"

#include <sofa/core/collision/ContactManager.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/map_ptr_stable_compare.h>
#include <vector>


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

protected:
    typedef sofa::helper::map_ptr_stable_compare<std::pair<core::CollisionModel*,core::CollisionModel*>,core::collision::Contact::SPtr> ContactMap;
    ContactMap contactMap;

public:
    Data<sofa::helper::OptionsGroup> response;
    Data<std::string> responseParams;
protected:
    DefaultContactManager();
    ~DefaultContactManager();
    void setContactTags(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::Contact::SPtr contact);

public:

    /// outputsVec fixes the reproducibility problems by storing contacts in the collision detection saved order
    /// if not given, it is still working but with eventual reproducibility problems
    void createContacts(const DetectionOutputMap& outputs);

    void init();
    void draw(const core::visual::VisualParams* vparams);

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

    void reset();
    void cleanup();

    virtual std::string getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2);

    /// virtual methods used for cleaning the pipeline after a dynamic graph node deletion.
    /**
     * Contacts can be attached to a deleted node and their deletion is a problem for the pipeline.
     * @param c is the list of deleted contacts.
     */
    virtual void removeContacts(const ContactVector &/*c*/);
    void setDefaultResponseType(const std::string &responseT)
    {
        if (response.getValue().size() == 0)
        {
            helper::vector<std::string> listResponse(1,responseT);

            sofa::helper::OptionsGroup responseOptions(listResponse);
            response.setValue(responseOptions);
        }
        else
        {
            sofa::helper::OptionsGroup* options = response.beginEdit();

            options->setSelectedItem(responseT);
            response.endEdit();
        }
    }

    std::string getDefaultResponseType() const { return response.getValue().getSelectedItem(); }

protected:
    static sofa::helper::OptionsGroup initializeResponseOptions(core::collision::Pipeline *pipeline);

    std::map<Instance,ContactMap> storedContactMap;

    virtual void changeInstance(Instance inst)
    {
        core::collision::ContactManager::changeInstance(inst);
        storedContactMap[instance].swap(contactMap);
        contactMap.swap(storedContactMap[inst]);
    }

    // count failure messages, so we don't continuously repeat them
    std::map<std::pair<std::string,std::pair<std::string,std::string> >, int> errorMsgCount;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
