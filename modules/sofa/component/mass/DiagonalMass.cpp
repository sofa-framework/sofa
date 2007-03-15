/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/mass/DiagonalMass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{
namespace component
{

namespace mass
{


// template<class Vec>
// void readVec1(Vec& vec, const char* str)
// {
// 	vec.clear();
// 	if (str==NULL) return;
// 	const char* str2 = NULL;
// 	for(;;)
// 	{
// 		double v = strtod(str,(char**)&str2);
// 		if (str2==str) break;
// 		str = str2;
// 		vec.push_back((typename Vec::value_type)v);
// 	}
// }



// template<class DataTypes, class MassType>
// void create(DiagonalMass<DataTypes, MassType>*& obj, simulation::tree::xml::ObjectDescription* arg)
// {
// 	simulation::tree::xml::createWithParent< DiagonalMass<DataTypes, MassType>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
// 	if (obj!=NULL)
// 	{
// 		if (arg->getAttribute("filename"))
// 		{
// 			obj->load(arg->getAttribute("filename"));
// 			arg->removeAttribute("filename");
// 		}
// 		obj->parseFields( arg->getAttributeMap() );
//
// 		if (arg->getAttribute("mass"))
// 		{
// 			std::vector<MassType> mass;
// 			readVec1(mass,arg->getAttribute("mass"));
// 			obj->clear();
// 			for (unsigned int i=0;i<mass.size();i++)
// 				obj->addMass(mass[i]);
// 		}
// 	}
// }




using namespace sofa::defaulttype;

template <>
double DiagonalMass<RigidTypes, RigidMass>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    const MassVector &masses= f_mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}
template <>
void DiagonalMass<RigidTypes, RigidMass>::init()
{
    ForceField<RigidTypes>::init();
}

template <>
void DiagonalMass<RigidTypes, RigidMass>::draw()
{
    const MassVector &masses= f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        const Quat& orient = x[i].getOrientation();
        //orient[3] = -orient[3];
        const RigidTypes::Vec3& center = x[i].getCenter();
        RigidTypes::Vec3 len;
        // The moment of inertia of a box is:
        //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
        //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
        //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
        // So to get lx,ly,lz back we need to do
        //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
        // Note that RigidMass inertiaMatrix is already divided by M
        double m00 = masses[i].inertiaMatrix[0][0];
        double m11 = masses[i].inertiaMatrix[1][1];
        double m22 = masses[i].inertiaMatrix[2][2];
        len[0] = sqrt(m11+m22-m00);
        len[1] = sqrt(m00+m22-m11);
        len[2] = sqrt(m00+m11-m22);

        helper::gl::Axis::draw(center, orient, len);
    }
}

SOFA_DECL_CLASS(DiagonalMass)

// Register in the Factory
int DiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
        .add< DiagonalMass<Vec3dTypes,double> >()
        .add< DiagonalMass<Vec3fTypes,float> >()
        .add< DiagonalMass<RigidTypes,RigidMass> >()
        ;

template class DiagonalMass<Vec3dTypes,double>;
template class DiagonalMass<Vec3fTypes,float>;
template class DiagonalMass<RigidTypes,RigidMass>;

} // namespace mass

} // namespace component

} // namespace sofa

