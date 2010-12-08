/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/vector.h>

#include <sofa/component/component.h>

namespace sofa
{
namespace core
{
namespace behavior
{
template< class T > class MechanicalState;

} // namespace behavior
} // namespace core
} // namespace sofa

namespace sofa
{

namespace component
{

namespace forcefield
{

/**
* @brief This class describes a simple elastic springs ForceField between DOFs positions and rest positions.
*
* Springs are applied to given degrees of freedom between their current positions and their rest shape positions.
* An external MechanicalState reference can also be passed to the ForceField as rest shape position.
*/
template<class DataTypes>
class RestShapeSpringsForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RestShapeSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef helper::vector< unsigned int > VecIndex;
    typedef helper::vector< Real >	 VecReal;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data< helper::vector< unsigned int > > points;
    Data< VecReal > stiffness;
    Data< VecReal > angularStiffness;
    Data< std::string > external_rest_shape;
    Data< helper::vector< unsigned int > > external_points;
    Data< bool > recompute_indices;

    sofa::core::behavior::MechanicalState< DataTypes > *restMState;

    VecDeriv Springs_dir;

    RestShapeSpringsForceField();

    /// BaseObject initialization method.
    void init();

    /// Add the forces.
    virtual void addForce(DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v, const core::MechanicalParams* mparams);

    /// Constant force has null variation.
    virtual void addDForce(DataVecDeriv& df, const DataVecDeriv& dx, const core::MechanicalParams* mparams);

    /// Brings ForceField contribution to the global system stiffness matrix.
    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const core::MechanicalParams* mparams );

    virtual void draw();
    bool addBBox(double* minBBox, double* maxBBox);

protected :

    VecIndex indices;
    VecReal k;
    VecIndex ext_indices;
    const VecCoord* pp_0;
private :

    bool useRestMState; /// An external MechanicalState is used as rest reference.
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_CPP)
#pragma warning(disable : 4231)

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec3dTypes>;
//extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec2dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec1dTypes>;
//extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec6dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Rigid3dTypes>;
//extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec3fTypes>;
//extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec2fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec1fTypes>;
//extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Vec6fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Rigid3fTypes>;
//extern template class SOFA_COMPONENT_FORCEFIELD_API RestShapeSpringsForceField<Rigid2fTypes>;
#endif

#endif // defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_H
