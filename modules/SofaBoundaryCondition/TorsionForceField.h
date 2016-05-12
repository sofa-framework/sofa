#ifndef TORSIONFORCEFIELD_H
#define TORSIONFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using sofa::defaulttype::Vec;
using sofa::defaulttype::Mat;
using sofa::core::behavior::ForceField;
using sofa::core::MechanicalParams;

#ifndef SOFA_DOUBLE
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::Rigid3fTypes;
#endif

#ifndef SOFA_FLOAT
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Rigid3dTypes;
#endif

template<typename DataTypes>
struct TorsionForceFieldTraits
{
	typedef typename DataTypes::Real Real;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
	typedef helper::vector<Coord> VecCoord;
	typedef helper::vector<Deriv> VecDeriv;
	enum { deriv_total_size = Coord::total_size };
	typedef Mat<deriv_total_size, deriv_total_size, Real> MatrixBlock;
};

//template<typename DataTypes>
//struct TorsionForceFieldUtility
//{
//	typedef typename DataTypes::Real Real;
//	typedef typename DataTypes::Coord Coord;
//	typedef typename DataTypes::Deriv Deriv;
//	enum { dim = Coord::spatial_dimensions };

//	static Vec<dim, Real> getCenter(const Coord& x) { return x; }
//	static Vec<dim, Real> getVCenter(const Deriv& v) { return v; }
//};


//template<class TCoord, class TDeriv, typename TReal>
//struct TorsionForceFieldUtility<defaulttype::StdVectorTypes<TCoord, TDeriv, TReal> >
//{
//	typedef defaulttype::StdVectorTypes<TCoord, TDeriv, TReal> DataType;
//	typedef typename DataType::Real Real;
//	enum { dim = DataType::Coord::spatial_dimensions };

//	static Vec<dim, Real>& getCenter(const typename DataType::Coord& x) { return x; }

//	static Vec<dim, Real>& getVCenter(const typename DataType::Deriv& v) { return v; }

//};

//template<int dim, typename TReal>
//struct TorsionForceFieldUtility<defaulttype::StdRigidTypes<dim, TReal> >
//{
//	typedef defaulttype::StdRigidTypes<dim, TReal> DataType;
//	typedef typename DataType::Real Real;
////	enum { dim = DataType::Coord::spatial_dimensions };

//	static Vec<dim, Real>& getCenter(const typename DataType::Coord& x) { return x.getCenter(); }
////	static const Vec<dim, Real>& getCenter(const DataType::Coord& x) { return x.getCenter(); }

//	static Vec<dim, Real> getVCenter(const typename DataType::Deriv& v) { return v.getVCenter(); }
////	static const Vec<dim, Real>& getVCenter(const DataType::Deriv& v) { return v.getVCenter(); }
//};

///
///	\brief TorsionForceField
///
///	This forcefield applies a torque to a set of selected nodes. The force is applied from a specified axis (origin and direction)
///	and a torque value.
///

template<typename DataTypes>
class TorsionForceField : public ForceField<DataTypes>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(TorsionForceField,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ForceField,DataTypes));

	typedef ForceField<DataTypes> Inherit;
	typedef TorsionForceFieldTraits<DataTypes> Traits;
	typedef typename Traits::Real Real;
	typedef typename Traits::Coord Coord;
	typedef typename Traits::Deriv Deriv;
	typedef typename Traits::VecCoord VecCoord;
	typedef typename Traits::VecDeriv VecDeriv;
	typedef typename Traits::MatrixBlock MatrixBlock;
	typedef typename DataTypes::CPos Pos;
//	typedef TorsionForceFieldUtility<DataTypes> FFUtil;
	typedef Data<VecCoord> DataVecCoord;
	typedef Data<VecDeriv> DataVecDeriv;

	typedef unsigned int PointId;
	typedef helper::vector<PointId> VecId;
	typedef Mat<3, 3, Real> Mat3;

public:
	TorsionForceField();
	virtual ~TorsionForceField();

	virtual void bwdInit();
	virtual void addForce(const MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v);
	virtual void addDForce(const MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx);
	virtual void addKToMatrix(defaulttype::BaseMatrix *matrix, double kFact, unsigned int &offset);

public :
	Data<VecId> m_indices;		///< indices of the selected nodes.
	Data<Real> m_torque;		///< torque to be applied.
	Data<Pos> m_axis;			///< direction of the axis.
	Data<Pos> m_origin;			///< origin of the axis.

protected :
	Pos m_u;					///< normalized axis
};

#ifndef SOFA_DOUBLE
template<>
void TorsionForceField<Rigid3fTypes>::addForce(const core::MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v);

template<>
void TorsionForceField<Rigid3fTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx);
#endif

#ifndef SOFA_FLOAT
template<>
void TorsionForceField<Rigid3dTypes>::addForce(const core::MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v);

template<>
void TorsionForceField<Rigid3dTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TORSIONFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class TorsionForceField<Vec3dTypes>;
extern template class TorsionForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class TorsionForceField<Vec3fTypes>;
extern template class TorsionForceField<Rigid3fTypes>;
#endif
#endif

} // namespace forcefield
} // namespace component
} // namespace sofa




#endif // TORSIONFORCEFIELD_H
