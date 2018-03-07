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
#ifndef SOFA_COMPONENT_COLLISION_TETRAHEDRONMODEL_H
#define SOFA_COMPONENT_COLLISION_TETRAHEDRONMODEL_H
#include "config.h"

#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <map>

namespace sofa
{

namespace component
{

namespace collision
{

class TetrahedronModel;

class Tetrahedron : public core::TCollisionElementIterator<TetrahedronModel>
{
public:
    Tetrahedron(TetrahedronModel* model, int index);
    Tetrahedron() {};
    explicit Tetrahedron(const core::CollisionElementIterator& i);

    const defaulttype::Vector3& p1() const;
    const defaulttype::Vector3& p2() const;
    const defaulttype::Vector3& p3() const;
    const defaulttype::Vector3& p4() const;
    int p1Index() const;
    int p2Index() const;
    int p3Index() const;
    int p4Index() const;

    const defaulttype::Vector3& p1Free() const;
    const defaulttype::Vector3& p2Free() const;
    const defaulttype::Vector3& p3Free() const;
    const defaulttype::Vector3& p4Free() const;

    const defaulttype::Vector3& v1() const;
    const defaulttype::Vector3& v2() const;
    const defaulttype::Vector3& v3() const;
    const defaulttype::Vector3& v4() const;

    defaulttype::Vector3 getBary(const defaulttype::Vector3& p) const;
    defaulttype::Vector3 getDBary(const defaulttype::Vector3& v) const;

    defaulttype::Vector3 getCoord(const defaulttype::Vector3& b) const;
    defaulttype::Vector3 getDCoord(const defaulttype::Vector3& b) const;

};

class SOFA_MISC_COLLISION_API TetrahedronModel : public core::CollisionModel
{
public:
    SOFA_CLASS(TetrahedronModel, core::CollisionModel);

    typedef defaulttype::Vec3Types InDataTypes;
    typedef defaulttype::Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Tetrahedron Element;
    friend class Tetrahedron;

protected:
    struct TetrahedronInfo
    {
        defaulttype::Vector3 coord0;
        defaulttype::Matrix3 coord2bary;
        defaulttype::Matrix3 bary2coord;
    };

    sofa::helper::vector<TetrahedronInfo> elems;
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra* tetra;

    core::behavior::MechanicalState<defaulttype::Vec3Types>* mstate;

    sofa::core::topology::BaseMeshTopology* _topology;

protected:

    TetrahedronModel();
public:
    virtual void init() override;

    // -- CollisionModel interface

    virtual void resize(int size) override;

    virtual void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(const core::visual::VisualParams*,int index) override;

    void draw(const core::visual::VisualParams* vparams) override;

    virtual void handleTopologyChange() override;

    core::behavior::MechanicalState<defaulttype::Vec3Types>* getMechanicalState() { return mstate; }

};

inline Tetrahedron::Tetrahedron(TetrahedronModel* model, int index)
    : core::TCollisionElementIterator<TetrahedronModel>(model, index)
{}

inline Tetrahedron::Tetrahedron(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<TetrahedronModel>(static_cast<TetrahedronModel*>(i.getCollisionModel()), i.getIndex())
{}

inline const defaulttype::Vector3& Tetrahedron::p1() const { return model->mstate->read(core::ConstVecCoordId::position())->getValue()[(*(model->tetra))[index][0]]; }
inline const defaulttype::Vector3& Tetrahedron::p2() const { return model->mstate->read(core::ConstVecCoordId::position())->getValue()[(*(model->tetra))[index][1]]; }
inline const defaulttype::Vector3& Tetrahedron::p3() const { return model->mstate->read(core::ConstVecCoordId::position())->getValue()[(*(model->tetra))[index][2]]; }
inline const defaulttype::Vector3& Tetrahedron::p4() const { return model->mstate->read(core::ConstVecCoordId::position())->getValue()[(*(model->tetra))[index][3]]; }

inline const defaulttype::Vector3& Tetrahedron::p1Free() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[(*(model->tetra))[index][0]]; }
inline const defaulttype::Vector3& Tetrahedron::p2Free() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[(*(model->tetra))[index][1]]; }
inline const defaulttype::Vector3& Tetrahedron::p3Free() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[(*(model->tetra))[index][2]]; }
inline const defaulttype::Vector3& Tetrahedron::p4Free() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[(*(model->tetra))[index][3]]; }

inline int Tetrahedron::p1Index() const { return (*(model->tetra))[index][0]; }
inline int Tetrahedron::p2Index() const { return (*(model->tetra))[index][1]; }
inline int Tetrahedron::p3Index() const { return (*(model->tetra))[index][2]; }
inline int Tetrahedron::p4Index() const { return (*(model->tetra))[index][3]; }

inline const defaulttype::Vector3& Tetrahedron::v1() const { return model->mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(model->tetra))[index][0]]; }
inline const defaulttype::Vector3& Tetrahedron::v2() const { return model->mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(model->tetra))[index][1]]; }
inline const defaulttype::Vector3& Tetrahedron::v3() const { return model->mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(model->tetra))[index][2]]; }
inline const defaulttype::Vector3& Tetrahedron::v4() const { return model->mstate->read(core::ConstVecDerivId::velocity())->getValue()[(*(model->tetra))[index][3]]; }

inline defaulttype::Vector3 Tetrahedron::getBary(const defaulttype::Vector3& p) const { return model->elems[index].coord2bary*(p-model->elems[index].coord0); }
inline defaulttype::Vector3 Tetrahedron::getDBary(const defaulttype::Vector3& v) const { return model->elems[index].coord2bary*(v); }
inline defaulttype::Vector3 Tetrahedron::getCoord(const defaulttype::Vector3& b) const { return model->elems[index].bary2coord*b + model->elems[index].coord0; }
inline defaulttype::Vector3 Tetrahedron::getDCoord(const defaulttype::Vector3& b) const { return model->elems[index].bary2coord*b; }

/// Mapper for TetrahedronModel
template<class DataTypes>
class ContactMapper<TetrahedronModel, DataTypes> : public BarycentricContactMapper<TetrahedronModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& P, int index, Real&)
    {
        Tetrahedron t(this->model, index);
        defaulttype::Vector3 b = t.getBary(P);
        return this->mapper->addPointInTetra(index, b.ptr());
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_TETRAHEDRONMODEL_CPP)
extern template class SOFA_MISC_COLLISION_API ContactMapper<TetrahedronModel, sofa::defaulttype::Vec3Types>;

#  ifdef _MSC_VER
// Manual declaration of non-specialized members, to avoid warnings from MSVC.
extern template SOFA_MISC_COLLISION_API void BarycentricContactMapper<TetrahedronModel, defaulttype::Vec3Types>::cleanup();
extern template SOFA_MISC_COLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<TetrahedronModel, defaulttype::Vec3Types>::createMapping(const char*);
#  endif
#endif



} // namespace collision

} // namespace component

} // namespace sofa

#endif
