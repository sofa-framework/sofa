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
#include <sofa/component/collision/response/contact/PenalityContactForceField.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::collision::response::contact
{

template<class DataTypes>
void PenalityContactForceField<DataTypes>::clear(sofa::Size reserve)
{
    prevContacts.swap(*contacts.beginEdit()); // save old contacts in prevContacts
    contacts.beginEdit()->clear();
    if (reserve)
        contacts.beginEdit()->reserve(reserve);
    contacts.endEdit();
}


template<class DataTypes>
void PenalityContactForceField<DataTypes>::addContact(sofa::Index m1, sofa::Index m2, sofa::Index index1, sofa::Index index2, const Deriv& norm, Real dist, Real ks, Real /*mu_s*/, Real /*mu_v*/, sofa::Index oldIndex)
{
    auto i = contacts.getValue().size();
    contacts.beginEdit()->resize(i+1);
    Contact& c = (*contacts.beginEdit())[i];
    c.m1 = m1;
    c.m2 = m2;
    c.index1 = index1;
    c.index2 = index2;
    c.norm = norm;
    c.dist = dist;
    c.ks = ks;
    c.pen = 0;
    if (oldIndex > 0 && oldIndex <= prevContacts.size())
    {
        c.age = prevContacts[oldIndex-1].age+1;
    }
    else
    {
        c.age = 0;
    }
    contacts.endEdit();
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& /*data_v1*/, const DataVecDeriv& /*data_v2*/ )
{
    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();

    type::vector<Contact>& cc = *contacts.beginEdit();

    f1.resize(x1.size());
    f2.resize(x2.size());

    for (sofa::Index i=0; i<cc.size(); i++)
    {
        Contact& c = cc[i];
        Coord u = x2[c.m2]-x1[c.m1];
        c.pen = c.dist - u*c.norm;
        if (c.pen > 0)
        {
            Real fN = c.ks * c.pen;
            Deriv force = -c.norm*fN;

            f1[c.m1]+=force;
            f2[c.m2]-=force;
        }
    }
    contacts.endEdit();

    data_f1.endEdit();
    data_f2.endEdit();

}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const type::vector<Contact>& cc = contacts.getValue();

    df1.resize(dx1.size());
    df2.resize(dx2.size());
    for (sofa::Index i=0; i<cc.size(); i++)
    {
        const Contact& c = cc[i];
        if (c.pen > 0) // + dpen > 0)
        {
            Coord du = dx2[c.m2]-dx1[c.m1];
            Real dpen = - du*c.norm;
            //if (c.pen < 0) dpen += c.pen; // start penality at distance 0
            Real dfN = c.ks * dpen * (Real)kFactor;
            Deriv dforce = -c.norm*dfN;
            df1[c.m1]+=dforce;
            df2[c.m2]-=dforce;
        }
    }

    data_df1.endEdit();
    data_df2.endEdit();

}

template <class DataTypes>
void PenalityContactForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams,
    const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    static constexpr auto N = DataTypes::spatial_dimensions;
    const Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,this->rayleighStiffness.getValue());

    const type::vector<Contact>& cc = contacts.getValue();

    if (this->mstate1 == this->mstate2)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat = matrix->getMatrix(this->mstate1);
        if (!mat) return;

        for (const auto& contact : cc)
        {
            if (contact.pen > 0)
            {
                const Real k = contact.ks * kFact;
                const sofa::Index p1 = mat.offset + Deriv::total_size * contact.m1;
                const sofa::Index p2 = mat.offset + Deriv::total_size * contact.m2;
                for(sofa::Index i = 0; i < N; ++i)
                {
                    for (sofa::Index j = 0; j < N; ++j)
                    {
                        const Real stiffness = k * contact.norm[i] * contact.norm[j];
                        mat.matrix->add(p1 + i, p1 + j, -stiffness);
                        mat.matrix->add(p1 + i, p2 + j,  stiffness);
                        mat.matrix->add(p2 + i, p1 + j,  stiffness);
                        mat.matrix->add(p2 + i, p2 + j, -stiffness);
                    }
                }
            }
        }
    }
    else
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat11 = matrix->getMatrix(this->mstate1);
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat22 = matrix->getMatrix(this->mstate2);
        sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat12 = matrix->getMatrix(this->mstate1, this->mstate2);
        sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat21 = matrix->getMatrix(this->mstate2, this->mstate1);

        if (!mat11 && !mat22 && !mat12 && !mat21) return;

        for (const auto& contact : cc)
        {
            if (contact.pen > 0)
            {
                const Real k = contact.ks * kFact;
                const sofa::Index p1 = Deriv::total_size * contact.m1;
                const sofa::Index p2 = Deriv::total_size * contact.m2;
                for(sofa::Index i = 0; i < N; ++i)
                {
                    for (sofa::Index j = 0; j < N; ++j)
                    {
                        const Real stiffness = k * contact.norm[i] * contact.norm[j];
                        mat11.matrix->add(mat11.offset + p1 + i, mat11.offset + p1 + j, -stiffness);
                        mat12.matrix->add(mat12.offRow + p1 + i, mat12.offCol + p2 + j,  stiffness);
                        mat21.matrix->add(mat21.offRow + p2 + i, mat21.offCol + p1 + j,  stiffness);
                        mat22.matrix->add(mat22.offset + p2 + i, mat22.offset + p2 + j, -stiffness);
                    }
                }
            }
        }
    }
}

template <class DataTypes>
void PenalityContactForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    const type::vector<Contact>& cc = contacts.getValue();

    if (this->mstate1 == this->mstate2)
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate1.get())
                           .withRespectToPositionsIn(this->mstate1.get());

        for (const auto& contact : cc)
        {
            if (contact.pen > 0)
            {
                const sofa::Index p1 = Deriv::total_size * contact.m1;
                const sofa::Index p2 = Deriv::total_size * contact.m2;
                const auto localMatrix = contact.ks * sofa::type::dyad(contact.norm, contact.norm);

                dfdx(p1, p1) += -localMatrix;
                dfdx(p1, p2) +=  localMatrix;
                dfdx(p2, p1) +=  localMatrix;
                dfdx(p2, p2) += -localMatrix;
            }
        }
    }
    else
    {
        auto* m1 = this->mstate1.get();
        auto* m2 = this->mstate2.get();

        auto df1_dx1 = matrix->getForceDerivativeIn(m1).withRespectToPositionsIn(m1);
        auto df1_dx2 = matrix->getForceDerivativeIn(m1).withRespectToPositionsIn(m2);
        auto df2_dx1 = matrix->getForceDerivativeIn(m2).withRespectToPositionsIn(m1);
        auto df2_dx2 = matrix->getForceDerivativeIn(m2).withRespectToPositionsIn(m2);

        df1_dx1.checkValidity(this);
        df1_dx2.checkValidity(this);
        df2_dx1.checkValidity(this);
        df2_dx2.checkValidity(this);

        for (const auto& contact : cc)
        {
            if (contact.pen > 0)
            {
                const sofa::Index p1 = Deriv::total_size * contact.m1;
                const sofa::Index p2 = Deriv::total_size * contact.m2;
                const auto localMatrix = contact.ks * sofa::type::dyad(contact.norm, contact.norm);

                df1_dx1(p1, p1) += -localMatrix;
                df1_dx2(p1, p2) +=  localMatrix;
                df2_dx1(p2, p1) +=  localMatrix;
                df2_dx2(p2, p2) += -localMatrix;
            }
        }
    }
}

template <class DataTypes>
void PenalityContactForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template <class DataTypes>
SReal PenalityContactForceField<DataTypes>::getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const
{
    msg_error() << "PenalityContactForceField::getPotentialEnergy-not-implemented !!!";
    return 0;
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) 
        return;
    
    using sofa::type::RGBAColor;

    const VecCoord& p1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();
    const type::vector<Contact>& cc = contacts.getValue();

    std::vector< type::Vec3 > points[4];

    for (sofa::Index i=0; i<cc.size(); i++)
    {
        const Contact& c = cc[i];
        Real d = c.dist - (p2[c.m2]-p1[c.m1])*c.norm;
        if (c.age > 10) //c.spen > c.mu_s * c.ks * 0.99)
            if (d > 0)
            {
                points[0].push_back(p1[c.m1]);
                points[0].push_back(p2[c.m2]);
            }
            else
            {
                points[1].push_back(p1[c.m1]);
                points[1].push_back(p2[c.m2]);
            }
        else if (d > 0)
        {
            points[2].push_back(p1[c.m1]);
            points[2].push_back(p2[c.m2]);
        }
        else
        {
            points[3].push_back(p1[c.m1]);
            points[3].push_back(p2[c.m2]);
        }
    }
    vparams->drawTool()->drawLines(points[0], 1, RGBAColor::magenta());
    vparams->drawTool()->drawLines(points[1], 1, RGBAColor::cyan());
    vparams->drawTool()->drawLines(points[2], 1, RGBAColor::red());
    vparams->drawTool()->drawLines(points[3], 1, RGBAColor::green());


    std::vector< type::Vec3 > pointsN;
    if (vparams->displayFlags().getShowNormals())
    {
        for (unsigned int i=0; i<cc.size(); i++)
        {
            const Contact& c = cc[i];
            Coord p = p1[c.m1] - c.norm;
            pointsN.push_back(p1[c.m1]);
            pointsN.push_back(p);


            p = p2[c.m2] + c.norm;
            pointsN.push_back(p2[c.m2]);
            pointsN.push_back(p);
        }
        vparams->drawTool()->drawLines(pointsN, 1, RGBAColor::yellow());
    }
}


template<class DataTypes>
void PenalityContactForceField<DataTypes>::grabPoint(
    const core::behavior::MechanicalState<defaulttype::Vec3Types> *tool,
    const type::vector< sofa::Index > &index,
    type::vector< std::pair< core::objectmodel::BaseObject*, type::Vec3f> > &result,
    type::vector< sofa::Index > &triangle,
    type::vector< sofa::Index > &index_point)
{
    const auto& contactsRef = contacts.getValue();

    if (static_cast< core::objectmodel::BaseObject *>(this->mstate1) == static_cast< const core::objectmodel::BaseObject *>(tool))
    {
        const auto& mstate2Pos = this->mstate2->read(core::ConstVecCoordId::position())->getValue();

        for (sofa::Index i=0; i< contactsRef.size(); i++)
        {
            for (sofa::Index j=0; j<index.size(); j++)
            {
                if (contactsRef[i].m1  == index[j])
                {
                    result.push_back(std::make_pair(static_cast< core::objectmodel::BaseObject *>(this),mstate2Pos[contactsRef[i].m2]));
                    triangle.push_back(contactsRef[i].index2);
                    index_point.push_back(index[j]);
                }
            }
        }
    }
    else if (static_cast< core::objectmodel::BaseObject *>(this->mstate2) == static_cast< const core::objectmodel::BaseObject *>(tool))
    {
        const auto& mstate1Pos = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
        for (sofa::Index i=0; i< contactsRef.size(); i++)
        {
            for (sofa::Index j=0; j<index.size(); j++)
            {
                if (contactsRef[i].m2  == index[j])
                {
                    result.push_back(std::make_pair(static_cast< core::objectmodel::BaseObject *>(this), mstate1Pos[contactsRef[i].m1]));
                    triangle.push_back(contactsRef[i].index1);
                    index_point.push_back(index[j]);
                }
            }
        }
    }


}

} // namespace sofa::component::collision::response::contact
