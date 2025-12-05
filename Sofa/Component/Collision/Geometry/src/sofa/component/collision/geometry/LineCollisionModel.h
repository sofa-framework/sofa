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
#include <sofa/core/fwd.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::collision::geometry
{

template<class DataTypes>
class LineCollisionModel;

template<class DataTypes>
class PointCollisionModel;

template<class TDataTypes>
class TLine : public core::TCollisionElementIterator<LineCollisionModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef LineCollisionModel<DataTypes> ParentModel;

    TLine(ParentModel* model, Index index);
    TLine() {}

    explicit TLine(const core::CollisionElementIterator& i);

    Index i1() const;
    Index i2() const;
    int flags() const;

    const Coord& p1() const;
    const Coord& p2() const;
    const Coord& p(Index i) const;

    const Coord& p1Free() const;
    const Coord& p2Free() const;

    const Deriv& v1() const;
    const Deriv& v2() const;

    Deriv n() const;

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;
};
using Line = TLine<sofa::defaulttype::Vec3Types>;

template<class TDataTypes>
class LineCollisionModel : public core::CollisionModel
{
public :
    SOFA_CLASS(SOFA_TEMPLATE(LineCollisionModel, TDataTypes), core::CollisionModel);

    enum LineFlag
    {
        FLAG_P1  = 1<<0, ///< Point 1  is attached to this line
        FLAG_P2  = 1<<1, ///< Point 2  is attached to this line
        FLAG_BP1 = 1<<2, ///< Point 1  is attached to this line and is a boundary
        FLAG_BP2 = 1<<3, ///< Point 2  is attached to this line and is a boundary
        FLAG_POINTS  = FLAG_P1|FLAG_P2,
        FLAG_BPOINTS = FLAG_BP1|FLAG_BP2,
    };

protected:
    struct LineData
    {
        sofa::Index p[2];
        // Triangles neighborhood
//		int tRight, tLeft;
    };

    sofa::type::vector<LineData> elems;
    bool needsUpdate;
    virtual void updateFromTopology();

    LineCollisionModel();

    void drawCollisionModel(const core::visual::VisualParams* vparams) override;

public:
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef LineCollisionModel<DataTypes> ParentModel;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef TLine<DataTypes> Element;
    static_assert(std::is_same_v<typename Element::Coord, Coord>, "Data mismatch");
    friend class TLine<DataTypes>;

    void init() override;

    // -- CollisionModel interface

    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    void computeContinuousBoundingTree(SReal dt, int maxDepth=0) override;

    void handleTopologyChange() override;

    bool canCollideWithElement(sofa::Index index, CollisionModel* model2, sofa::Index index2) override;

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return mstate; }

    Deriv velocity(sofa::Index index)const;

    virtual sofa::Index getElemEdgeIndex(sofa::Index index) const { return index; }

    int getLineFlags(sofa::Index i);

    Data<bool> d_bothSide; ///< activate collision on both side of the line model (when surface normals are defined on these lines)

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

    sofa::core::topology::BaseMeshTopology* getCollisionTopology() override
    {
        return l_topology.get();
    }

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    Data<bool> d_displayFreePosition; ///< Display Collision Model Points free position(in green)

    /// Link to be set to the topology container in the component graph.
    SingleLink<LineCollisionModel<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    core::behavior::MechanicalState<DataTypes>* mstate;
    Topology* topology;
    PointCollisionModel<sofa::defaulttype::Vec3Types>* mpoints;
    int meshRevision;
};

template<class DataTypes>
inline TLine<DataTypes>::TLine(ParentModel* model, Index index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{
}

template<class DataTypes>
inline TLine<DataTypes>::TLine(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}

#if !defined(SOFA_COMPONENT_COLLISION_LINECOLLISIONMODEL_CPP)
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API TLine<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API LineCollisionModel<sofa::defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::collision::geometry
