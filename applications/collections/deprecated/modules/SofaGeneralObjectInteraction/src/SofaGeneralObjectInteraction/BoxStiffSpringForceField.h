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
#include <SofaGeneralObjectInteraction/config.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/engine/select/BoxROI.h>
#include <sofa/component/engine/select/NearestPointROI.h>

namespace sofa::component::interactionforcefield
{

/** Set springs between the particles located inside a given box.
*/
template <class DataTypes>
class BoxStiffSpringForceField : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BoxStiffSpringForceField, DataTypes), sofa::core::objectmodel::BaseObject);

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);

        MechanicalState* mstate1 = nullptr;
        MechanicalState* mstate2 = nullptr;
        std::string object1 = arg->getAttribute("object1","@./");
        std::string object2 = arg->getAttribute("object2","@./");

        context->findLinkDest(mstate1, object1, nullptr);
        context->findLinkDest(mstate2, object2, nullptr);

        sofa::type::fixed_array<typename engine::select::BoxROI<DataTypes>::SPtr, 2> boxes;
        for (const auto* mstate : {mstate1, mstate2})
        {
            auto boxROI = sofa::core::objectmodel::New<engine::select::BoxROI<DataTypes> >();
            const std::size_t id = mstate != mstate1;
            boxes[id] = boxROI;
            boxROI->setName("box_" + mstate->getName());
            boxROI->d_X0.setParent(mstate->findData("position"));
            if (arg)
            {
                const std::string boxString = arg->getAttribute("box_object" + std::to_string(id+1), "");
                boxROI->d_alignedBoxes.read(boxString);
            }
            if (context)
            {
                context->addObject(boxROI);
            }
        }

        auto np = sofa::core::objectmodel::New<sofa::component::engine::select::NearestPointROI<DataTypes> >(mstate1, mstate2);
        np->f_radius.setValue(std::numeric_limits<typename DataTypes::Real>::max());
        np->setName(helper::NameDecoder::shortName(np->getClassName()));
        if (context)
        {
            context->addObject(np);
        }

        np->d_inputIndices1.setParent(&boxes[0]->d_indices);
        np->d_inputIndices2.setParent(&boxes[1]->d_indices);

        auto springs = sofa::core::objectmodel::New<sofa::component::solidmechanics::spring::StiffSpringForceField<DataTypes> >(mstate1, mstate2);
        springs->d_indices1.setParent(&np->f_indices1);
        springs->d_indices2.setParent(&np->f_indices2);
        springs->d_lengths.setParent(&np->d_distances);
        if (arg)
        {
            springs->parse(arg);
        }
        if (context)
        {
            context->addObject(springs);
        }

        return obj;
    }

};

#if  !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_CPP)
extern template class SOFA_SOFAGENERALOBJECTINTERACTION_API BoxStiffSpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_SOFAGENERALOBJECTINTERACTION_API BoxStiffSpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_SOFAGENERALOBJECTINTERACTION_API BoxStiffSpringForceField<defaulttype::Vec1Types>;
extern template class SOFA_SOFAGENERALOBJECTINTERACTION_API BoxStiffSpringForceField<defaulttype::Vec6Types>;

#endif

} //namespace sofa::component::interactionforcefield
