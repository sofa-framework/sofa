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
#ifndef SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_H
#define SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_H

#include <Sofa.Component.PhysicalModel.SolidMechanics.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/TopologyData.inl>


namespace sofa
{

namespace component
{

namespace physicalmodel
{

namespace solidmechanics
{

namespace fem
{

/** Compute Finite Element forces based on 6D beam elements.
*/
template<class DataTypes>
class SOFA_SOLIDMECHANICS_API BeamFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BeamFEMForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

	using Real = typename DataTypes::Real;
	using Coord = typename DataTypes::Coord;
	using Deriv = typename DataTypes::Deriv;
	using VecCoord = typename DataTypes::VecCoord;
	using VecDeriv = typename DataTypes::VecDeriv;
	using VecReal = typename DataTypes::VecReal;
	using DataVecCoord = Data<VecCoord>;
	using DataVecDeriv = Data<VecDeriv>;

	using Index = unsigned int;
	using Edge = core::topology::BaseMeshTopology::Edge ;
	using VecEdges = sofa::helper::vector<core::topology::BaseMeshTopology::Edge> ;
	using VecIndex = helper::vector<Index> ;

	using Quat = helper::Quater<Real>;
	using Vec3 = defaulttype::Vec<3, Real> ;
	using Displacement = defaulttype::Vec<12, Real> ; ///< the displacement vector
	using Transformation = defaulttype::Mat<3, 3, Real>; ///< matrix for rigid transformations like rotations
	using StiffnessMatrix = defaulttype::Mat<12, 12, Real>;


	Data< bool> _useSymmetricAssembly; ///< use symmetric assembly of the matrix K

	/// Per edge inforation
	topology::EdgeData< sofa::helper::vector<Real>> d_youngModulus; ///< Young Modulus (E)
	topology::EdgeData< sofa::helper::vector<Real>> d_poissonRatio; ///< Poisson ratio (nu)
	topology::EdgeData< sofa::helper::vector<Real>> d_r; ///< radius of the section
	topology::EdgeData< sofa::helper::vector<Real>> d_innerR; ///< inner radius
	topology::EdgeData< sofa::helper::vector<Real>> d_L; ///< Length

private:
	sofa::helper::vector<Real> _G; //shear modulus
	sofa::helper::vector<Real> _Iy;
	sofa::helper::vector<Real> _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
	sofa::helper::vector<Real> _J;  //Polar moment of inertia (J = Iy + Iz)
	sofa::helper::vector<Real> _A; // A is the cross-sectional area;
	sofa::helper::vector<Real> _Asy; //_Asy is the y-direction effective shear area =  10/9 (for solid circular section) or 0 for a non-Timoshenko beam
	sofa::helper::vector<Real> _Asz; //_Asz is the z-direction effective shear area;
	sofa::helper::vector<StiffnessMatrix> _k_loc;
	sofa::helper::vector<Quat> _quat;

	const Real _epsilon;

    sofa::core::topology::BaseMeshTopology* _topology;

    BeamFEMForceField();
    virtual ~BeamFEMForceField();

public:
    virtual void init() override;
    virtual void bwdInit() override;
    virtual void reinit() override;

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    virtual void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;
	virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;

    void draw(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

	/// Pre-construction check method called by ObjectFactory.
	/// Check that DataTypes matches the MechanicalState.
	template<class T>
	static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
	{
		if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL && context->getMechanicalState() != NULL)
			return false;

		return core::objectmodel::BaseObject::canCreate(obj, context, arg);
	}

	template<class T>
	static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
	{
		typename T::SPtr obj;

		if( context)
		{
			obj = sofa::core::objectmodel::New<T>();
			context->addObject(obj);
		}

		if (arg) obj->parse(arg);

		return obj;
	}

private:
	void initInternalData();
	void reinitBeam(std::size_t i);
	void computeStiffness(std::size_t i);

	void drawElement(std::size_t i, Index a, Index b, std::array<std::vector<defaulttype::Vector3>, 3>& points, const VecCoord& x);

    ////////////// large displacements method
	void initLarge(std::size_t i, Index a, Index b);
	void accumulateForceLarge( VecDeriv& f, const VecCoord& x, std::size_t i, Index a, Index b);
	void applyStiffnessLarge(VecDeriv& f, const VecDeriv& x, std::size_t i, Index a, Index b, Real fact=1.0);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_SIMPLE_FEM_API BeamFEMForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_SIMPLE_FEM_API BeamFEMForceField<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace fem

} // namespace solidmechanics

} // namespace physicalmodel

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_PHYSICALMODEL_SOLIDMECHANICS_FEM_BEAMFEMFORCEFIELD_H
