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
#include <sofa/gui/component/performer/SuturePointPerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/collision/geometry/TriangleModel.h>


namespace sofa::gui::component::performer
{

template <class DataTypes>
SuturePointPerformer<DataTypes>::SuturePointPerformer(BaseMouseInteractor *i)
    :TInteractionPerformer<DataTypes>(i)
    , first(1)
    , fixedIndex(0)
    , SpringObject(nullptr)
    ,FixObject(nullptr) {}


template <class DataTypes>
void SuturePointPerformer<DataTypes>::start()
{
    if (first) //first click
    {
        const BodyPicked picked = this->m_interactor->getBodyPicked();
        const auto* CollisionModel = dynamic_cast<sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>* >(picked.body);

        if (picked.body == nullptr || CollisionModel == nullptr)
        {
            msg_error(this->m_interactor) << "No picked body in first clic.";
            return;
        }

        firstPicked = this->m_interactor->getBodyPicked();
        first = false;
    }
    else // second click
    {
        const BodyPicked picked = this->m_interactor->getBodyPicked();
        auto* CollisionModel = dynamic_cast<sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>* >(picked.body);

        if (picked.body == nullptr || CollisionModel == nullptr)
        {
            msg_error(this->m_interactor) << "No picked body in second clic.";
            return;
        }

        CollisionModel->getContext()->get (SpringObject, sofa::core::objectmodel::BaseContext::SearchRoot);

        sofa::core::topology::BaseMeshTopology* triangleContainer;
        CollisionModel->getContext()->get (triangleContainer);

        sofa::component::statecontainer::MechanicalObject <defaulttype::Vec3Types>* MechanicalObject;
        CollisionModel->getContext()->get (MechanicalObject,  sofa::core::objectmodel::BaseContext::SearchRoot);

        CollisionModel->getContext()->get (FixObject, sofa::core::objectmodel::BaseContext::SearchRoot);

        if (!SpringObject)
        {
            msg_error(this->m_interactor) << "Can't find SpringForceField.";
            return;
        }
        else if (!triangleContainer)
        {
            msg_error(this->m_interactor) << "Can't find a topology.";
            return;
        }
        else if (triangleContainer->getTriangles().empty())
        {
            msg_error(this->m_interactor) << "Can't find a topology with triangles.";
            return;
        }
        else if (!MechanicalObject)
        {
            msg_error(this->m_interactor) << "Can't find MechanicalObject.";
            return;
        }
        else if (!FixObject)
        {
            msg_error(this->m_interactor) << "Can't find FixObject.";
            return;
        }


        // Get vertices of both triangles
        sofa::type::vector<sofa::type::Vec3 > listCoords;
        const core::topology::BaseMeshTopology::Triangle Triangle1 = triangleContainer->getTriangle(firstPicked.indexCollisionElement);
        const core::topology::BaseMeshTopology::Triangle Triangle2 = triangleContainer->getTriangle(picked.indexCollisionElement);

        for (unsigned int i=0; i<3; i++)
        {
            const sofa::type::Vec3& tmp = (MechanicalObject->read(core::vec_id::read_access::position)->getValue())[ Triangle1[i] ];
            listCoords.push_back (tmp);
        }

        for (unsigned int i=0; i<3; i++)
        {
            const sofa::type::Vec3& tmp = (MechanicalObject->read(core::vec_id::read_access::position)->getValue())[ Triangle2[i] ];
            listCoords.push_back (tmp);
        }

        sofa::type::vector<unsigned int> pointToSuture;
        pointToSuture.resize(2);

        sofa::type::vector<sofa::type::Vec3 > listPoint;
        listPoint.push_back(firstPicked.point);
        listPoint.push_back(picked.point);

        // Find the closest dof to pickedPoint:
        for (unsigned int tri = 0; tri<2; ++tri)
        {
            double distance1 = 0.0;
            double distance2 = 0.0;
            double sum = 0.0;
            unsigned int cpt = 0;

            //iter 1;
            for (unsigned int i =0; i<3; i++)
                sum += (listCoords[tri*3][i] - listPoint[tri][i])*(listCoords[tri*3][i] - listPoint[tri][i]);

            for (unsigned int i =1; i<3; i++)
            {
                sum = 0.0;
                for (unsigned int j=0; j<3; j++)
                    sum += (listCoords[i+tri*3][j] - listPoint[tri][j])*(listCoords[i+tri*3][j] - listPoint[tri][j]);

                distance2 = sqrt (sum);

                if (distance2 < distance1) // this point is closer
                {
                    cpt = i+tri*3;
                    distance1 = distance2;
                }
            }

            pointToSuture[tri] = cpt;
        }

        // in "world" referentiel
        pointToSuture[0] = Triangle1[ pointToSuture[0] ];
        pointToSuture[1] = Triangle2[ pointToSuture[1] ];
        fixedIndex = pointToSuture[1];

        // Create fixConstraint
        FixObject->addConstraint(fixedIndex);

        // Create spring
        SpringObject->addSpring(pointToSuture[0], pointToSuture[1], stiffness, damping, 0);

        // Saving springs
        Spring newSpring;
        newSpring.m1 = pointToSuture[0];
        newSpring.m2 = pointToSuture[1];
        newSpring.ks = 100000;
        newSpring.kd = (Real)damping;
        newSpring.initpos = 0;

        addedSprings.push_back(newSpring);
        first = true;
    }

}

template <class DataTypes>
SuturePointPerformer<DataTypes>::~SuturePointPerformer()
{
    if (SpringObject) //means we added a spring
    {
        sofa::type::vector<Spring> vecSprings = SpringObject->getSprings();
        const unsigned int nbr = vecSprings.size();

        for (unsigned int i = 0; i<addedSprings.size(); ++i)
            SpringObject->removeSpring(nbr-1-i );

        for (unsigned int i = 0; i<addedSprings.size(); ++i)
            SpringObject->addSpring(addedSprings[i].m1, addedSprings[i].m2, addedSprings[i].ks, addedSprings[i].kd, addedSprings[i].initpos);
    }

    if (FixObject)
        FixObject->removeConstraint(fixedIndex);
}

} // namespace sofa::gui::component::performer
