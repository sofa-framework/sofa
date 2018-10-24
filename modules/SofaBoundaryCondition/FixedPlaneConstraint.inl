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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_INL

#include <SofaBoundaryCondition/FixedPlaneConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using sofa::helper::WriteAccessor;
using sofa::defaulttype::Vec;
using sofa::component::topology::PointSubsetData;
using sofa::component::topology::TopologySubsetDataHandler;


/////////////////////////// DEFINITION OF FCPointHandler (INNER CLASS) /////////////////////////////
template <class DataTypes>
class FixedPlaneConstraint<DataTypes>::FCPointHandler :
        public TopologySubsetDataHandler<BaseMeshTopology::Point, SetIndexArray >
{
public:
    typedef typename FixedPlaneConstraint<DataTypes>::SetIndexArray SetIndexArray;

    FCPointHandler(FixedPlaneConstraint<DataTypes>* _fc, PointSubsetData<SetIndexArray>* _data)
        : TopologySubsetDataHandler<BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}

    void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);

    bool applyTestCreateFunction(unsigned int /*index*/,
                                 const helper::vector< unsigned int > & /*ancestors*/,
                                 const helper::vector< double > & /*coefs*/);
protected:
    FixedPlaneConstraint<DataTypes> *fc;
};

/// Define RemovalFunction
template< class DataTypes>
void FixedPlaneConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}

/// Define TestNewPointFunction
template< class DataTypes>
bool FixedPlaneConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const helper::vector<unsigned int> &, const helper::vector<double> &)
{
    if (fc)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/////////////////////////// DEFINITION OF FixedPlaneConstraint /////////////////////////////////////
template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint()
    : d_direction( initData(&d_direction,"direction","normal direction of the plane"))
    , d_dmin( initData(&d_dmin,(Real)0,"dmin","Minimum plane distance from the origin"))
    , d_dmax( initData(&d_dmax,(Real)0,"dmax","Maximum plane distance from the origin") )
    , d_indices( initData(&d_indices,"indices","Indices of the fixed points"))
{
    m_selectVerticesFromPlanes=false;
    m_pointHandler = new FCPointHandler(this, &d_indices);
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::~FixedPlaneConstraint()
{
    if (m_pointHandler)
        delete m_pointHandler;
}

/// Matrix Integration interface
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::applyConstraint(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix)
{
    MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(mstate.get(mparams));
    if(r)
    {
        /// Implement plane constraint only when the direction is along the coordinates directions
        // TODO : generalize projection to any direction

        const unsigned int N = Deriv::size();
        Coord dir=d_direction.getValue();
        for (auto& index : d_indices.getValue())
        {
            /// Reset Fixed Row and Col
            for (unsigned int c=0; c<N; ++c)
                if (dir[c]!=0.0)
                    r.matrix->clearRowCol(r.offset + N * index + c);
            /// Set Fixed Vertex
            for (unsigned int c=0; c<N; ++c)
                if (dir[c]!=0.0)
                    r.matrix->set(r.offset + N * index + c, r.offset + N * index + c, 1.0);
        }
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::applyConstraint(const MechanicalParams* mparams,
                                                      BaseVector* vect,
                                                      const MultiMatrixAccessor* matrix)
{
    int o = matrix->getGlobalOffset(mstate.get(mparams));
    if (o >= 0)
    {
        unsigned int offset = (unsigned int)o;
        /// Implement plane constraint only when the direction is along the coordinates directions
        // TODO : generalize projection to any direction
        Coord dir=d_direction.getValue();

        const unsigned int N = Deriv::size();

        for (auto& index : d_indices.getValue())
        {
            for (unsigned int c=0; c<N; ++c)
                if (dir[c]!=0.0)
                    vect->clear(offset + N * index + c);
        }
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::addConstraint(int index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::removeConstraint(int index)
{
    removeValue(*d_indices.beginEdit(),(unsigned int)index);
    d_indices.endEdit();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectResponse(const MechanicalParams* mparams, DataVecDeriv& resData)
{
    WriteAccessor<DataVecDeriv> res = resData;
    projectResponseImpl(mparams, res.wref());
}

/// project dx to constrained space (dx models a velocity)
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectVelocity(const MechanicalParams* /*mparams*/, DataVecDeriv& /*vData*/)
{

}

/// project x to constrained space (x models a position)
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectPosition(const MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{

}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned /*offset*/ )
{
    /// clears the rows and columns associated with constrained particles
    unsigned blockSize = DataTypes::deriv_total_size;

    for(auto& index : d_indices.getValue())
    {
        M->clearRowsCols((index) * blockSize,(index+1) * (blockSize) );
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectJacobianMatrix(const MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    WriteAccessor<DataMatrixDeriv> c = cData;
    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseImpl(mparams, rowIt.row());
        ++rowIt;
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDirection(Coord dir)
{
    if (dir.norm2()>0)
    {
        d_direction.setValue(dir);
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::selectVerticesAlongPlane()
{
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
    for(unsigned int i=0; i<x.size(); ++i)
    {
        if (isPointInPlane(x[i]))
            addConstraint(i);
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::init()
{
    ProjectiveConstraintSet<DataTypes>::init();

    m_topology = getContext()->getMeshTopology();

    /// test that dmin or dmax are different from zero
    if (d_dmin.getValue()!=d_dmax.getValue())
        m_selectVerticesFromPlanes=true;

    if (m_selectVerticesFromPlanes)
        selectVerticesAlongPlane();

    /// Initialize functions and parameters
    d_indices.createTopologicalEngine(m_topology, m_pointHandler);
    d_indices.registerTopologicalData();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDminAndDmax(const Real _dmin,const Real _dmax)
{
    d_dmin=_dmin;
    d_dmax=_dmax;
    m_selectVerticesFromPlanes=true;
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::draw(const VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
    vparams->drawTool()->disableLighting();

    helper::vector<sofa::defaulttype::Vector3> points;
    for(auto& index : d_indices.getValue())
    {
        points.push_back({x[index][0], x[index][1], x[index][2]});
    }

    vparams->drawTool()->drawPoints(points, 10, {1,1.0,0.5,1});
}

/// This function are there to provide kind of type translation to the vector one so we can
/// implement the algorithm as is the different objects where of similar type.
/// this solution is not really satisfactory but for the moment it does the job.
/// A better solution would that all the used types are following the same iterface which
/// requires to touch core sofa classes.
sofa::defaulttype::Vec3d& getVec(sofa::defaulttype::Rigid3dTypes::Deriv& i){ return i.getVCenter(); }
sofa::defaulttype::Vec3d& getVec(sofa::defaulttype::Rigid3dTypes::Coord& i){ return i.getCenter(); }
const sofa::defaulttype::Vec3d& getVec(const sofa::defaulttype::Rigid3dTypes::Coord& i){ return i.getCenter(); }
sofa::defaulttype::Vec3d& getVec(sofa::defaulttype::Vec3dTypes::Deriv& i){ return i; }
const sofa::defaulttype::Vec3d& getVec(const sofa::defaulttype::Vec3dTypes::Deriv& i){ return i; }
sofa::defaulttype::Vec6d& getVec(sofa::defaulttype::Vec6dTypes::Deriv& i){ return i; }
const sofa::defaulttype::Vec6d& getVec(const sofa::defaulttype::Vec6dTypes::Deriv& i){ return i; }

sofa::defaulttype::Vec3f& getVec(sofa::defaulttype::Rigid3fTypes::Deriv& i){ return i.getVCenter(); }
sofa::defaulttype::Vec3f& getVec(sofa::defaulttype::Rigid3fTypes::Coord& i){ return i.getCenter(); }
const sofa::defaulttype::Vec3f& getVec(const sofa::defaulttype::Rigid3fTypes::Coord& i){ return i.getCenter(); }
sofa::defaulttype::Vec3f& getVec(sofa::defaulttype::Vec3fTypes::Deriv& i){ return i; }
const sofa::defaulttype::Vec3f& getVec(const sofa::defaulttype::Vec3fTypes::Deriv& i){ return i; }
sofa::defaulttype::Vec6f& getVec(sofa::defaulttype::Vec6fTypes::Deriv& i){ return i; }
const sofa::defaulttype::Vec6f& getVec(const sofa::defaulttype::Vec6fTypes::Deriv& i){ return i; }

template<class DataTypes>
bool FixedPlaneConstraint<DataTypes>::isPointInPlane(Coord p) const
{
    Vec<Coord::spatial_dimensions,Real> pos = getVec(p) ;
    Real d=pos*getVec(d_direction.getValue());
    if ((d>d_dmin.getValue())&& (d<d_dmax.getValue()))
        return true;
    else
        return false;
}

template <class DataTypes>
template <class T>
void FixedPlaneConstraint<DataTypes>::projectResponseImpl(const MechanicalParams* mparams, T& res) const
{
    SOFA_UNUSED(mparams);

    Coord dir=d_direction.getValue();
    for (auto& index : d_indices.getValue())
    {
        /// only constraint one projection of the displacement to be zero
        getVec(res[index]) -= getVec(dir) * dot( getVec(res[index]), getVec(dir));
    }
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
