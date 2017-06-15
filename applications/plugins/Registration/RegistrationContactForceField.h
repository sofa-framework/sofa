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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_REGISTRATIONCONTACTFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_REGISTRATIONCONTACTFORCEFIELD_H

#include <Registration/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{


template<class DataTypes>
class RegistrationContactForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
  SOFA_CLASS(SOFA_TEMPLATE(RegistrationContactForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

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
protected:
	
	class Contact
	{
	public:
	  
	  int m1, m2;   ///< the two extremities of the spring: masses m1 and m2 
	  int index1, index2; ///< the index of the two collision elements	  
	  Deriv norm;   ///< contact normal, from m1 to m2
	  Real dist;    ///< minimum distance between the points
	  Real ks;      ///< spring stiffness
	  Real mu_s;    ///< coulomb friction coefficient (currently unused)
	  Real mu_v;    ///< viscous friction coefficient
	  Real pen;     ///< current penetration
	  int age;      ///< how old is this contact


		Contact(int _m1=0, int _m2=0, int _index1=0, int _index2=0, Deriv _norm=Deriv(), Real _dist=(Real)0, Real _ks=(Real)0, Real _mu_s=(Real)0, Real _mu_v=(Real)0, Real _pen=(Real)0, int _age=0)
	    : m1(_m1),m2(_m2),index1(_index1),index2(_index2),norm(_norm),dist(_dist),ks(_ks),mu_s(_mu_s),mu_v(_mu_v),pen(_pen),age(_age)
	    {
	    }


        inline friend std::istream& operator >> ( std::istream& in, Contact& c ){
			in>>c.m1>>c.m2>>c.index1>>c.index2>>c.norm>>c.dist>>c.ks>>c.mu_s>>c.mu_v>>c.pen>>c.age;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Contact& c ){
			out << c.m1<< " " <<c.m2<< " " << c.index1<< " " <<c.index2<< " " <<c.norm<< " " <<c.dist<<" " <<c.ks<<" " <<c.mu_s<<" " <<c.mu_v<<" " <<c.pen<<" " <<c.age;
			return out;
        }
	};
	
	Data<sofa::helper::vector<Contact> > contacts;
	
	// contacts from previous frame
    sofa::helper::vector<Contact> prevContacts;


public:

	RegistrationContactForceField(MechanicalState* object1, MechanicalState* object2)
	  : Inherit(object1, object2), contacts(initData(&contacts,"contacts", "Contacts"))
	{
	}
	
	RegistrationContactForceField()
	{
	}
	
	void clear(int reserve = 0);

	void addContact(int m1, int m2, int index1, int index2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f, int oldIndex = 0);
	
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 );

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2);

    virtual SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const ;

    const helper::vector< Contact >& getContact() const { return contacts.getValue();}
	 
	// -- tool grabing utility
	void grabPoint( const core::behavior::MechanicalState<defaulttype::Vec3Types> *tool, 
					const helper::vector< unsigned int > &index, 
                    helper::vector< std::pair< core::objectmodel::BaseObject*, defaulttype::Vec3f> > &result, 
					helper::vector< unsigned int > &triangle,
					helper::vector< unsigned int > &index_point) ;

	void draw(const core::visual::VisualParams* vparams);

    void updateForceMask();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_REGISTRATIONCONTACTFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_REGISTRATION_API RegistrationContactForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_REGISTRATION_API RegistrationContactForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_REGISTRATIONCONTACTFORCEFIELD_H */
