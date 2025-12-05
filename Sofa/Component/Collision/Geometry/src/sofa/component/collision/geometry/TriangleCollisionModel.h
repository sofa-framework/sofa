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
#include <sofa/component/collision/geometry/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/fwd.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/VecId.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::component::collision::geometry
{

template<class DataTypes>
class TriangleCollisionModel;

template<class DataTypes>
class PointCollisionModel;


template<class TDataTypes>
class TTriangle : public core::TCollisionElementIterator< TriangleCollisionModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef TriangleCollisionModel<DataTypes> ParentModel;
	typedef typename DataTypes::Real Real;

    using Index = sofa::Index;

    TTriangle(ParentModel* model, Index index);
    TTriangle() {}
    explicit TTriangle(const core::CollisionElementIterator& i);
	TTriangle(ParentModel* model, Index index, helper::ReadAccessor<typename DataTypes::VecCoord>& /*x*/);

    const Coord& p1() const;
    const Coord& p2() const;
    const Coord& p3() const;

    const Coord& p(Index i)const;

    Index p1Index() const;
    Index p2Index() const;
    Index p3Index() const;

    const Coord& p1Free() const;
    const Coord& p2Free() const;
    const Coord& p3Free() const;

    const Coord& operator[](Index i) const;

    const Deriv& v1() const;
    const Deriv& v2() const;
    const Deriv& v3() const;
    const Deriv& v(Index i) const;


    const Deriv& n() const;
    Deriv& n();

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;

    int flags() const;

	TTriangle& shape() { return *this; }
    const TTriangle& shape() const { return *this; }

    Coord interpX(type::Vec<2,Real> bary) const
    {
		return (p1()*(1-bary[0]-bary[1])) + (p2()*bary[0]) + (p3()*bary[1]);
	}
};
using Triangle = TTriangle<sofa::defaulttype::Vec3Types>;

/**
 * This class will create collision elements based on a triangle and/or quad mesh.
 * It uses directly the information of the topology and the dof to compute the triangle normals, BB and BoundingTree.
 * The class \sa TTriangle is used to access specific triangle of this collision Model.
 */
template<class TDataTypes>
class TriangleCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangleCollisionModel, TDataTypes), core::CollisionModel);

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef TTriangle<DataTypes> Element;
    static_assert(std::is_same_v<typename Element::Coord, Coord>, "Data mismatch");
    friend class TTriangle<DataTypes>;

    enum TriangleFlag
    {
        FLAG_P1  = 1<<0, ///< Point 1  is attached to this triangle
        FLAG_P2  = 1<<1, ///< Point 2  is attached to this triangle
        FLAG_P3  = 1<<2, ///< Point 3  is attached to this triangle
        FLAG_E23 = 1<<3, ///< Edge 2-3 is attached to this triangle
        FLAG_E31 = 1<<4, ///< Edge 3-1 is attached to this triangle
        FLAG_E12 = 1<<5, ///< Edge 1-2 is attached to this triangle
        FLAG_BE23 = 1<<6, ///< Edge 2-3 is attached to this triangle and is a boundary
        FLAG_BE31 = 1<<7, ///< Edge 3-1 is attached to this triangle and is a boundary
        FLAG_BE12 = 1<<8, ///< Edge 1-2 is attached to this triangle and is a boundary
        FLAG_POINTS  = FLAG_P1|FLAG_P2|FLAG_P3,
        FLAG_EDGES   = FLAG_E12|FLAG_E23|FLAG_E31,
        FLAG_BEDGES  = FLAG_BE12|FLAG_BE23|FLAG_BE31,
    };

	enum { NBARY = 2 };

    Data<bool> d_bothSide; ///< activate collision on both side of the triangle model
    Data<bool> d_computeNormals; ///< set to false to disable computation of triangles normal
    Data<bool> d_useCurvature; ///< use the curvature of the mesh to avoid some self-intersection test
    
    /// Link to be set to the topology container in the component graph.
    SingleLink<TriangleCollisionModel<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    core::behavior::MechanicalState<DataTypes>* m_mstate; ///< Pointer to the corresponding MechanicalState
    sofa::core::topology::BaseMeshTopology* m_topology; ///< Pointer to the corresponding Topology

    VecDeriv m_normals; ///< Vector of normal direction per triangle.

    /** Pointer to the triangle array of this collision model.
     * Will point directly to the topology triangle buffer if only triangles are present. If topology is using/mixing quads and triangles,
     * This pointer will target \sa m_internalTriangles
     * @brief m_triangles
     */
    const sofa::core::topology::BaseMeshTopology::SeqTriangles* m_triangles;

    sofa::core::topology::BaseMeshTopology::SeqTriangles m_internalTriangles; ///< Internal Buffer of triangles to combine quads splitted and other triangles.

    bool m_needsUpdate; ///< parameter storing the info boundingTree has to be recomputed.
    int m_topologyRevision; ///< internal revision number to check if topology has changed.

    PointCollisionModel<sofa::defaulttype::Vec3Types>* m_pointModels;

protected:

    TriangleCollisionModel();

    virtual void updateFromTopology();
    virtual void updateNormals();
    void drawCollisionModel(const core::visual::VisualParams* vparams) override;

public:
    void init() override;

    // -- CollisionModel interface

    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    void computeContinuousBoundingTree(SReal dt, int maxDepth=0) override;

    void draw(const core::visual::VisualParams*, sofa::Index index) override;

    bool canCollideWithElement(sofa::Index index, CollisionModel* model2, sofa::Index index2) override;

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return m_mstate; }
    const core::behavior::MechanicalState<DataTypes>* getMechanicalState() const { return m_mstate; }

    const VecCoord& getX() const { return(getMechanicalState()->read(core::vec_id::read_access::position)->getValue()); }
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& getTriangles() const { return *m_triangles; }
    const VecDeriv& getNormals() const { return m_normals; }
    int getTriangleFlags(sofa::core::topology::BaseMeshTopology::TriangleID i);

    Deriv velocity(sofa::Index index)const;


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
        {
            arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                          "' found in the context node.");
            return false;
        }
        return BaseObject::canCreate(obj, context, arg);
    }

    void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

    sofa::core::topology::BaseMeshTopology* getCollisionTopology() override
    {
        return l_topology.get();
    }
};

template<class DataTypes>
inline TTriangle<DataTypes>::TTriangle(ParentModel* model, Index index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TTriangle<DataTypes>::TTriangle(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{}

template<class DataTypes>
inline TTriangle<DataTypes>::TTriangle(ParentModel* model, Index index, helper::ReadAccessor<typename DataTypes::VecCoord>& x)
    : TTriangle(model, index)
{
    SOFA_UNUSED(x);
}

#if !defined(SOFA_COMPONENT_COLLISION_TRIANGLECOLLISIONMODEL_CPP)
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API TTriangle<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API TriangleCollisionModel<defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::collision::geometry
