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
#ifndef SOFA_COMPONENT_CONSTRAINT_ATTACHCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_ATTACHCONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/PairInteractionConstraint.inl>
#include <sofa/component/constraint/AttachConstraint.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

#include <sofa/component/topology/PointSubset.h>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace core::componentmodel::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::componentmodel::behavior;

template<>
inline void AttachConstraint<defaulttype::Rigid3dTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations)
{
    x2.getCenter() = x1.getCenter();
    if (!freeRotations)
        x2.getOrientation() = x1.getOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid3fTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations)
{
    x2.getCenter() = x1.getCenter();
    if (!freeRotations)
        x2.getOrientation() = x1.getOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid2dTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations)
{
    x2.getCenter() = x1.getCenter();
    if (!freeRotations)
        x2.getOrientation() = x1.getOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid2fTypes>::projectPosition(Coord& x1, Coord& x2, bool freeRotations)
{
    x2.getCenter() = x1.getCenter();
    if (!freeRotations)
        x2.getOrientation() = x1.getOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid3dTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations)
{
    x2.getVCenter() = x1.getVCenter();
    if (!freeRotations)
        x2.getVOrientation() = x1.getVOrientation();
    //x2 = Deriv();
}

template<>
inline void AttachConstraint<defaulttype::Rigid3fTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations)
{
    x2.getVCenter() = x1.getVCenter();
    if (!freeRotations)
        x2.getVOrientation() = x1.getVOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid2dTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations)
{
    x2.getVCenter() = x1.getVCenter();
    if (!freeRotations)
        x2.getVOrientation() = x1.getVOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid2fTypes>::projectVelocity(Deriv& x1, Deriv& x2, bool freeRotations)
{
    x2.getVCenter() = x1.getVCenter();
    if (!freeRotations)
        x2.getVOrientation() = x1.getVOrientation();
}

template<>
inline void AttachConstraint<defaulttype::Rigid3dTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool oneway)
{
    if (oneway)
    {
        if (!freeRotations)
            dx2 = Deriv();
        else
            dx2.getVCenter().clear();
    }
    else
    {
        if (!freeRotations)
        {
            dx1 += dx2;
            dx2 = dx1;
        }
        else
        {
            dx1.getVCenter() += dx2.getVCenter();
            dx2.getVCenter() = dx1.getVCenter();
        }
    }
}

template<>
inline void AttachConstraint<defaulttype::Rigid3fTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool oneway)
{
    if (oneway)
    {
        if (!freeRotations)
            dx2 = Deriv();
        else
            dx2.getVCenter().clear();
    }
    else
    {
        if (!freeRotations)
        {
            dx1 += dx2;
            dx2 = dx1;
        }
        else
        {
            dx1.getVCenter() += dx2.getVCenter();
            dx2.getVCenter() = dx1.getVCenter();
        }
    }
}

template<>
inline void AttachConstraint<defaulttype::Rigid2dTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool oneway)
{
    if (oneway)
    {
        if (!freeRotations)
            dx2 = Deriv();
        else
            dx2.getVCenter().clear();
    }
    else
    {
        if (!freeRotations)
        {
            dx1 += dx2;
            dx2 = dx1;
        }
        else
        {
            dx1.getVCenter() += dx2.getVCenter();
            dx2.getVCenter() = dx1.getVCenter();
        }
    }
}

template<>
inline void AttachConstraint<defaulttype::Rigid2fTypes>::projectResponse(Deriv& dx1, Deriv& dx2, bool freeRotations, bool oneway)
{
    if (oneway)
    {
        if (!freeRotations)
            dx2 = Deriv();
        else
            dx2.getVCenter().clear();
    }
    else
    {
        if (!freeRotations)
        {
            dx1 += dx2;
            dx2 = dx1;
        }
        else
        {
            dx1.getVCenter() += dx2.getVCenter();
            dx2.getVCenter() = dx1.getVCenter();
        }
    }
}

template<>
inline unsigned int AttachConstraint<defaulttype::Rigid3dTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::static_size;
    else
        return Deriv::size();
}

template<>
inline unsigned int AttachConstraint<defaulttype::Rigid3fTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::static_size;
    else
        return Deriv::size();
}

template<>
inline unsigned int AttachConstraint<defaulttype::Rigid2dTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::static_size;
    else
        return Deriv::size();
}

template<>
inline unsigned int AttachConstraint<defaulttype::Rigid2fTypes>::DerivConstrainedSize(bool freeRotations)
{
    if (freeRotations)
        return Deriv::static_size;
    else
        return Deriv::size();
}

#if 0
// Define TestNewPointFunction
template< class DataTypes>
bool AttachConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    AttachConstraint<DataTypes> *fc= (AttachConstraint<DataTypes> *)param;
    if (fc)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Define RemovalFunction
template< class DataTypes>
void AttachConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    AttachConstraint<DataTypes> *fc= (AttachConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}
#endif

template <class DataTypes>
AttachConstraint<DataTypes>::AttachConstraint()
    : f_indices1( initData(&f_indices1,"indices1","Indices of the source points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the fixed points on the second model") )
    , f_radius( initData(&f_radius,(Real)-1,"radius", "Radius to search corresponding fixed point if no indices are given") )
    , f_twoWay( initData(&f_twoWay,false,"twoWay", "true if forces should be projected back from model2 to model1") )
    , f_freeRotations( initData(&f_freeRotations,false,"freeRotations", "true to keep rotations free (only used for Rigid DOFs)") )
    , f_lastPos( initData(&f_lastPos,"lastPos", "position at which the attach constraint should become inactive") )
    , f_lastDir( initData(&f_lastDir,"lastDir", "direction from lastPos at which the attach coustraint should become inactive") )
    , f_clamp( initData(&f_clamp, false,"clamp", "true to clamp particles at lastPos instead of freeing them.") )
{
    // default to indice 0
//     f_indices1.beginEdit()->push_back(0);
//     f_indices1.endEdit();
//     f_indices2.beginEdit()->push_back(0);
//     f_indices2.endEdit();
}

#if 0
// Handle topological changes
template <class DataTypes> void AttachConstraint<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
}
#endif

template <class DataTypes>
AttachConstraint<DataTypes>::~AttachConstraint()
{
}

template <class DataTypes>
void AttachConstraint<DataTypes>::clearConstraints()
{
    f_indices1.beginEdit()->clear();
    f_indices1.endEdit();
    f_indices2.beginEdit()->clear();
    f_indices2.endEdit();
}

template <class DataTypes>
void AttachConstraint<DataTypes>::addConstraint(unsigned int index1, unsigned int index2)
{
    f_indices1.beginEdit()->push_back(index1);
    f_indices1.endEdit();
    f_indices2.beginEdit()->push_back(index2);
    f_indices2.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void AttachConstraint<DataTypes>::init()
{
    this->core::componentmodel::behavior::PairInteractionConstraint<DataTypes>::init();
    if (f_radius.getValue() >= 0 && f_indices1.getValue().size()==0 && f_indices2.getValue().size()==0 && this->mstate1 && this->mstate2)
    {
        const Real maxR = f_radius.getValue();
        VecCoord& x1 = *this->mstate1->getX();
        VecCoord& x2 = *this->mstate2->getX();
        for (unsigned int i2=0; i2<x2.size(); ++i2)
        {
            int best = -1;
            Real bestR = maxR;
            for (unsigned int i1=0; i1<x1.size(); ++i1)
            {
                Real r = (x2[i2]-x1[i1]).norm();
                if (r <= bestR)
                {
                    best = i1;
                    bestR = r;
                }
            }
            if (best >= 0)
            {
                addConstraint(best, i2);
            }
        }
    }
#if 0
    // Initialize functions and parameters
    topology::PointSubset my_subset = f_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );
#endif
    activeFlags.resize(f_indices2.getValue().size());
    std::fill(activeFlags.begin(), activeFlags.end(), true);
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectPosition(VecCoord& res1, VecCoord& res2)
{
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    const bool freeRotations = f_freeRotations.getValue();
    const bool last = (f_lastDir.isSet() && f_lastDir.getValue().norm() > 1.0e-10);
    const bool clamp = f_clamp.getValue();
    const bool log = this->f_printLog.getValue();
    activeFlags.resize(indices2.size());
    //std::cout << "lastDir="<<f_lastDir.getValue()<<std::endl;
    //std::cout << "lastPos="<<f_lastPos.getValue()<<std::endl;
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;
        Coord p = res1[indices1[i]];
        if (last)
        {
            defaulttype::Vec3d p3d;
            DataTypes::get(p3d[0],p3d[1],p3d[2],p);
            if ((p3d-f_lastPos.getValue())*f_lastDir.getValue() > 0.0)
            {
                if (clamp)
                {
                    if (activeFlags[i])
                        std::cout << "AttachConstraint: point "<<indices1[i]<<" stopped."<<std::endl;
                    DataTypes::set(p,f_lastPos.getValue()[0],f_lastPos.getValue()[1],f_lastPos.getValue()[2]);
                    //p = f_lastPos.getValue();
                }
                else
                {
                    if (activeFlags[i])
                        std::cout << "AttachConstraint: point "<<indices1[i]<<" is free."<<std::endl;
                }
                active = false;
            }
        }
        activeFlags[i] = active;
        if (active || clamp)
        {
            if (log) std::cout << "AttachConstraint: x2["<<indices2[i]<<"] = x1["<<indices1[i]<<"]\n";
            //res2[indices2[i]] = res1[indices1[i]];
            projectPosition(p, res2[indices2[i]], freeRotations);
        }
    }
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectVelocity(VecDeriv& res1, VecDeriv& res2)
{
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    const bool freeRotations = f_freeRotations.getValue();
    const bool clamp = f_clamp.getValue();
    const bool log = this->f_printLog.getValue();
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;
        if (i < activeFlags.size()) active = activeFlags[i];
        //res2[indices2[i]] = res1[indices1[i]];
        if (active)
        {
            if (log) std::cout << "AttachConstraint: v2["<<indices2[i]<<"] = v1["<<indices1[i]<<"]\n";
            projectVelocity(res1[indices1[i]], res2[indices2[i]], freeRotations);
        }
        else if (clamp)
        {
            if (log) std::cout << "AttachConstraint: v2["<<indices2[i]<<"] = 0\n";
            Deriv v = Deriv();
            projectVelocity(v, res2[indices2[i]], freeRotations);
        }
    }
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectResponse(VecDeriv& res1, VecDeriv& res2)
{
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    const bool twoway = f_twoWay.getValue();
    const bool freeRotations = f_freeRotations.getValue();
    const bool clamp = f_clamp.getValue();
    const bool log = this->f_printLog.getValue();
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        bool active = true;
        if (i < activeFlags.size()) active = activeFlags[i];
        if (active)
        {
            if (log)
            {
                if (twoway) std::cout << "AttachConstraint: r2["<<indices2[i]<<"] = r1["<<indices2[i]<<"] = (r2["<<indices2[i]<<"] + r2["<<indices2[i]<<"])\n";
                else        std::cout << "AttachConstraint: r2["<<indices2[i]<<"] = 0\n";
            }
            projectResponse(res1[indices1[i]], res2[indices2[i]], freeRotations, twoway);
        }
        else if (clamp)
        {
            if (log) std::cout << "AttachConstraint: r2["<<indices2[i]<<"] = 0\n";
            Deriv v = Deriv();
            projectResponse(v, res2[indices2[i]], freeRotations, false);
        }
    }
}

// Matrix Integration interface
template <class DataTypes>
void AttachConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int &offset)
{
    //std::cout << "applyConstraint in Matrix with offset = " << offset << std::endl;
    const SetIndexArray & indices = f_indices2.getValue().getArray();
    const unsigned int N = Deriv::size();
    const unsigned int NC = DerivConstrainedSize(f_freeRotations.getValue());
    unsigned int i=0;
    const bool clamp = f_clamp.getValue();
    const bool log = this->f_printLog.getValue();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++i)
    {
        if (!clamp && i < activeFlags.size() && !activeFlags[i]) continue;
        if (log) std::cout << "AttachConstraint: apply in matrix column/row "<<(*it)<<"\n";
        // Reset Fixed Row and Col
        for (unsigned int c=0; c<NC; ++c)
            mat->clearRowCol(offset + N * (*it) + c);
        // Set Fixed Vertex
        for (unsigned int c=0; c<NC; ++c)
            mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0e10);
    }
}

template <class DataTypes>
void AttachConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int &offset)
{
    //std::cout << "applyConstraint in Vector with offset = " << offset << std::endl;

    const SetIndexArray & indices = f_indices2.getValue().getArray();
    const unsigned int N = Deriv::size();
    const unsigned int NC = DerivConstrainedSize(f_freeRotations.getValue());
    unsigned int i=0;
    const bool clamp = f_clamp.getValue();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++i)
    {
        if (!clamp && i < activeFlags.size() && !activeFlags[i]) continue;
        for (unsigned int c=0; c<NC; ++c)
            vect->clear(offset + N * (*it) + c);
    }
}


template <class DataTypes>
void AttachConstraint<DataTypes>::draw()
{
    if (!getContext()->
        getShowBehaviorModels()) return;
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    VecCoord& x1 = *this->mstate1->getX();
    VecCoord& x2 = *this->mstate2->getX();
    glDisable (GL_LIGHTING);
    glPointSize(5);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (activeFlags.size() > i && !activeFlags[i]) continue;
        gl::glVertexT(x2[indices2[i]]);
    }
    glEnd();
    glPointSize(1);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (activeFlags.size() > i && !activeFlags[i]) continue;
        gl::glVertexT(x1[indices1[i]]);
        gl::glVertexT(x2[indices2[i]]);
    }
    glEnd();
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
