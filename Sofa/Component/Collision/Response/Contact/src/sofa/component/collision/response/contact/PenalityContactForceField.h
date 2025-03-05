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

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>

namespace sofa::component::collision::response::contact
{

template <class T>
class PenalityContact
{
public:
    //using Coord = T::Coord;
    //using Deriv = T::Deriv;
    //using Real = Coord::value_type;
    typedef typename T::Coord Coord;
    typedef typename T::Deriv Deriv;
    typedef typename Coord::value_type Real;

    sofa::Index m1, m2;         ///< the indices of the vertices the force is applied to
    sofa::Index index1, index2; ///< the indices of the two collision elements (currently unused)
    Deriv norm;         ///< contact normal, from m1 to m2
    Real dist;          ///< distance threshold below which a repulsion force is applied
    Real ks;            ///< spring stiffness
    Real pen;           ///< current penetration depth
    int age;            ///< how old is this contact

    PenalityContact(sofa::Index _m1 = 0, sofa::Index _m2 = 0, sofa::Index _index1 = 0, sofa::Index _index2 = 0, Deriv _norm = Deriv(), Real _dist = (Real)0, Real _ks = (Real)0, Real /*_mu_s*/ = (Real)0, Real /*_mu_v*/ = (Real)0, Real _pen = (Real)0, int _age = 0)
        : m1(_m1), m2(_m2), index1(_index1), index2(_index2), norm(_norm), dist(_dist), ks(_ks),/*mu_s(_mu_s),mu_v(_mu_v),*/pen(_pen), age(_age)
    {
    }

    inline friend std::istream& operator >> (std::istream& in, PenalityContact& c)
    {
        in >> c.m1 >> c.m2 >> c.index1 >> c.index2 >> c.norm >> c.dist >> c.ks >>/*c.mu_s>>c.mu_v>>*/c.pen >> c.age;
        return in;
    }

    inline friend std::ostream& operator << (std::ostream& out, const PenalityContact& c)
    {
        out << c.m1 << " " << c.m2 << " " << c.index1 << " " << c.index2 << " " << c.norm << " " << c.dist << " " << c.ks << " " <</*c.mu_s<<" " <<c.mu_v<<" " <<*/c.pen << " " << c.age;
        return out;
    }
};


/** Distance-based, frictionless penalty force. The force is applied to vertices attached to collision elements.
  */
template<class DataTypes>
class PenalityContactForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PenalityContactForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef DataTypes DataTypes1;
    typedef DataTypes DataTypes2;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    using Contact = PenalityContact<DataTypes>;

    Data<sofa::type::vector<Contact> > contacts; ///< Contacts

protected:
    // contacts from previous frame
    sofa::type::vector<Contact> prevContacts;


    PenalityContactForceField(MechanicalState* object1, MechanicalState* object2)
        : Inherit(object1, object2), contacts(initData(&contacts,"contacts", "Contacts"))
    {
    }

    PenalityContactForceField()
    {
    }

public:
    void clear(sofa::Size reserve = 0);

    void addContact(sofa::Index m1, sofa::Index m2, sofa::Index index1, sofa::Index index2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f, sofa::Index oldIndex = 0);

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;

    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;

    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const override;

    const type::vector< Contact >& getContact() const { return contacts.getValue();}

    // -- tool grabbing utility
    void grabPoint( const core::behavior::MechanicalState<defaulttype::Vec3Types> *tool,
            const type::vector< sofa::Index > &index,
            type::vector< std::pair< core::objectmodel::BaseObject*, type::Vec3f> > &result,
            type::vector< sofa::Index > &triangle,
            type::vector< sofa::Index > &index_point) ;

    void draw(const core::visual::VisualParams* vparams) override;

};

#if !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_PENALITYCONTACTFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API PenalityContactForceField<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::collision::response::contact
