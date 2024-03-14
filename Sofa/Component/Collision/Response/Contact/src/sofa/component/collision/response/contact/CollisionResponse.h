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
#include <sofa/component/collision/response/contact/config.h>

#include <sofa/core/collision/ContactManager.h>
#include <sofa/simulation/fwd.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/map_ptr_stable_compare.h>


namespace sofa::component::collision::response::contact
{

class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API CollisionResponse : public core::collision::ContactManager
{
public :
    SOFA_CLASS(CollisionResponse,sofa::core::collision::ContactManager);

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
            sofa::helper::OptionsGroup options = initializeResponseOptions(context);
            obj->response.setValue(options);
        }

        if (arg)
            obj->parse(arg);

        //SOFA_ATTRIBUTE_DISABLED("v21.12 (PR#2522)", "v24.06","This attribute was only added to build a compatibility layer on the response name.")
        {
            static const std::map<std::string,std::string> renamingResponseMethod = {
                {"ray", "RayContact"},
                {"default", "PenalityContactForceField"},
                {"FrictionContact", "FrictionContactConstraint"},
                {"registration", "RegistrationContactForceField"},
                {"stick", "StickContactForceField"},
            };

            const std::string responseDesired = arg->getAttribute("response","");
            const auto it = renamingResponseMethod.find(responseDesired);

            if(it != renamingResponseMethod.end())
            {
                msg_error("CollisionResponse")
                    << "Option \"" << it->first << "\" "
                    << "for data \"response\" has been renamed since v21.12 (PR#2522). "
                    << "Use \"" << it->second << "\" instead.";
            }
        }
        
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
    typedef sofa::helper::map_ptr_stable_compare<
                /* key */  std::pair<core::CollisionModel*, core::CollisionModel*>,
                /* value */core::collision::Contact::SPtr
            > ContactMap;

    CollisionResponse();
    ~CollisionResponse() override = default;

    static void setContactTags(core::CollisionModel* model1, core::CollisionModel* model2,
                        core::collision::Contact::SPtr contact);

    ContactMap contactMap;
    std::map<Instance,ContactMap> storedContactMap;

    void changeInstance(Instance inst) override ;

    static sofa::helper::OptionsGroup initializeResponseOptions(sofa::core::objectmodel::BaseContext *pipeline);

    // count failure messages, so we don't continuously repeat them
    std::map<std::pair<std::string,std::pair<std::string,std::string> >, int> errorMsgCount;

    /// Write messages in stringstream in case a contact failed to be created
    void contactCreationError(std::stringstream &errorStream, const core::CollisionModel *model1,
                              const core::CollisionModel *model2, std::string &responseUsed);

    /// Create contacts if it has not been created before
    void createNewContacts(const DetectionOutputMap &outputsMap, Size &nbContact);

    void removeInactiveContacts(const DetectionOutputMap &outputsMap, Size& nbContact);

    /// compute and set the number of contacts attached to each collision model
    /// The number of contacts corresponds to the number of collision models
    /// currently in contact with a collision model.
    void setNumberOfContacts() const;

};


} // namespace sofa::component::collision::response::contact
