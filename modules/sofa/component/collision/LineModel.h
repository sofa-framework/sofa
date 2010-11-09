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
#ifndef SOFA_COMPONENT_COLLISION_LINEMODEL_H
#define SOFA_COMPONENT_COLLISION_LINEMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/collision/LocalMinDistanceFilter.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/collision/PointModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class LineModel;
class LineLocalMinDistanceFilter;

class Line : public core::TCollisionElementIterator<LineModel>
{
public:
    Line(LineModel* model, int index);

    explicit Line(core::CollisionElementIterator& i);

    unsigned i1() const;
    unsigned i2() const;

    const Vector3& p1() const;
    const Vector3& p2() const;

    const Vector3& p1Free() const;
    const Vector3& p2Free() const;

    const Vector3& v1() const;
    const Vector3& v2() const;

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;

    bool activated(core::CollisionModel *cm = 0) const;

    // Return respectively the Vertex composing the neighbor Rigt and Left Triangle
//	const Vector3* tRight() const;
//	const Vector3* tLeft() const;
};

class LineActiver
{
public:
    LineActiver() {}
    virtual ~LineActiver() {}
    virtual bool activeLine(int /*index*/, core::CollisionModel * /*cm*/ = 0) {return true;}
};

class SOFA_COMPONENT_COLLISION_API LineModel : public core::CollisionModel
{
public :
    SOFA_CLASS(LineModel,sofa::core::CollisionModel);
protected:
    struct LineData
    {
        int i1,i2;
        // Triangles neighborhood
//		int tRight, tLeft;
    };

    sofa::helper::vector<LineData> elems;
    bool needsUpdate;
    virtual void updateFromTopology();


public:
    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Line Element;

    friend class Line;

    LineModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    void draw();

    virtual void handleTopologyChange();

    bool canCollideWithElement(int index, CollisionModel* model2, int index2);

    core::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }

    //virtual const char* getTypeName() const { return "Line"; }

    LineLocalMinDistanceFilter *getFilter() const;

    //template< class TFilter >
    //TFilter *getFilter() const
    //{
    //	if (m_lmdFilter != 0)
    //		return m_lmdFilter;
    //	else
    //		return &m_emptyFilter;
    //}

    void setFilter(LineLocalMinDistanceFilter * /*lmdFilter*/);

    Data<bool> bothSide; // to activate collision on both-side of the both side of the line model (when surface normals are defined on these lines)


protected:

    core::behavior::MechanicalState<Vec3Types>* mstate;
    Topology* topology;
    PointModel* mpoints;
    int meshRevision;
    LineLocalMinDistanceFilter *m_lmdFilter;

    Data<std::string> LineActiverEngine;

    LineActiver *myActiver;


};

inline Line::Line(LineModel* model, int index)
    : core::TCollisionElementIterator<LineModel>(model, index)
{
//	activated = model->myActiver->activeLine(index);
}

inline Line::Line(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<LineModel>(static_cast<LineModel*>(i.getCollisionModel()), i.getIndex())
{
//	LineModel* CM = static_cast<LineModel*>(i.getCollisionModel());
//	activated = CM->myActiver->activeLine(i.getIndex());
}

inline unsigned Line::i1() const { return model->elems[index].i1; }
inline unsigned Line::i2() const { return model->elems[index].i2; }

inline const Vector3& Line::p1() const { return (*model->mstate->getX())[model->elems[index].i1]; }
inline const Vector3& Line::p2() const { return (*model->mstate->getX())[model->elems[index].i2]; }

inline const Vector3& Line::p1Free() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[model->elems[index].i1]; }
inline const Vector3& Line::p2Free() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[model->elems[index].i2]; }

inline const Vector3& Line::v1() const { return (*model->mstate->getV())[model->elems[index].i1]; }
inline const Vector3& Line::v2() const { return (*model->mstate->getV())[model->elems[index].i2]; }

inline bool Line::hasFreePosition() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->isSet(); }

inline bool Line::activated(core::CollisionModel *cm) const
{
    return model->myActiver->activeLine(index, cm);
}

//inline const Vector3* Line::tRight() const {
//	if (model->elems[index].tRight != -1)
//		return &(*model->mstate->getX())[model->elems[index].tRight];
//	else
//		return NULL;
//}

//inline const Vector3* Line::tLeft() const {
//	if (model->elems[index].tLeft != -1)
//		return &(*model->mstate->getX())[model->elems[index].tLeft];
//	else
//		return NULL;
//}

//class LineMeshModel : public LineModel
//{
//protected:
//    int meshRevision;
//    void updateFromTopology();
//
//public:
//    typedef topology::MeshTopology Topology;
//
//    LineMeshModel();
//
//    virtual void init();
//
//    Topology* getMeshTopology() { return mesh; }
//
//protected:
//    Topology* mesh;
//};

//class LineSetModel : public LineModel
//{
//public:
//
//    LineSetModel();
//
//    ///\Todo
//    virtual void init();
//
//    Topology* getMeshTopology() { return mesh; }
//
//protected:
//
//    Topology* mesh;
//
//    void updateFromTopology();
//};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
