/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_BEAMMASSMATRIX_H
#define SOFA_COMPONENT_MASS_BEAMMASSMATRIX_H

#include <Mass.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <sofa/core/behavior/Mass.h>

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class TMassType>
class BeamMassMatrix : public core::behavior::Mass<DataTypes>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE2(BeamMassMatrix,DataTypes,TMassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

	using Inherited = core::behavior::Mass<DataTypes>;
	using Real = typename DataTypes::Real;
	using Deriv = typename DataTypes::Deriv;
	using Coord = typename DataTypes::Coord;
	using VecDeriv = typename DataTypes::VecDeriv;

	using VecCoord = typename DataTypes::VecCoord;
	using DataVecCoord = core::objectmodel::Data<VecCoord>;
	using DataVecDeriv = core::objectmodel::Data<VecDeriv>;

	using Edge = core::topology::BaseMeshTopology::Edge;

	using MassType = TMassType;

	// In case of non 3D template
	using Vec3 = defaulttype::Vec<3,Real>;
	/// assumes the geometry object type is 3D
	using GeometricalTypes = defaulttype::StdVectorTypes< Vec3, Vec3, Real >  ;

	/// Mass info are stocked on vertices and edges (if lumped matrix)
	//topology::PointData<helper::vector<MassType> >  vertexMassInfo;

	/// the mass density used to compute the mass from a mesh topology and geometry
	Data<sofa::helper::vector<Real>> m_massDensity;

	topology::EdgeData< sofa::helper::vector<Real>> d_r; ///< radius of the section
	topology::EdgeData< sofa::helper::vector<Real>> d_innerR; ///< inner radius
	topology::EdgeData< sofa::helper::vector<Real>> d_L; ///< Length

private:
	sofa::helper::vector<Real> _Iy;
	sofa::helper::vector<Real> _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
	sofa::helper::vector<Real> _J;  //Polar moment of inertia (J = Iy + Iz)
	sofa::helper::vector<Real> _A; // A is the cross-sectional area;

	sofa::helper::vector<Real> M00;
	sofa::helper::vector<Real> M11;
	sofa::helper::vector<Real> M22;
	sofa::helper::vector<Real> M33;
	sofa::helper::vector<Real> M44;
	sofa::helper::vector<Real> M55;
	sofa::helper::vector<Real> M66;
	sofa::helper::vector<Real> M77;
	sofa::helper::vector<Real> M88;
	sofa::helper::vector<Real> M99;
	sofa::helper::vector<Real> M1010;
	sofa::helper::vector<Real> M1111;
	sofa::helper::vector<Real> M24;
	sofa::helper::vector<Real> M15;
	sofa::helper::vector<Real> M06;
	sofa::helper::vector<Real> M17;
	sofa::helper::vector<Real> M57;
	sofa::helper::vector<Real> M28;
	sofa::helper::vector<Real> M48;
	sofa::helper::vector<Real> M39;
	sofa::helper::vector<Real> M210;
	sofa::helper::vector<Real> M410;
	sofa::helper::vector<Real> M810;
	sofa::helper::vector<Real> M111;
	sofa::helper::vector<Real> M511;
	sofa::helper::vector<Real> M711;

protected:

	BeamMassMatrix();
	~BeamMassMatrix();

	void massInitialization();

private:
	sofa::core::topology::BaseMeshTopology* _topology;
	sofa::component::topology::EdgeSetGeometryAlgorithms<GeometricalTypes>* edgeGeo;


public:
	virtual void clear();
	virtual void init() override;
	virtual void reinit() override;

	virtual void addMDx(const core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;

	virtual void addForce(const core::MechanicalParams*, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MASS_BEAMMASSMATRIX_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MASS_API BeamMassMatrix<defaulttype::Rigid3dTypes, double>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MASS_API BeamMassMatrix<defaulttype::Rigid3fTypes, float>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MASS_BEAMMASSMATRIX_H
