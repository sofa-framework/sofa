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
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/engine/analyze/ShapeMatching.h>
#include <sofa/helper/decompose.h>
#include <iostream>
#include <sofa/helper/IndexOpenMP.h>
#include <sofa/type/Mat.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace sofa::component::engine::analyze
{
template<class Real>
inline const type::Vec<3,Real>& center(const type::Vec<3,Real>& c)
{
    return c;
}

template<class _Real>
inline type::Vec<3,_Real>& center(type::Vec<3,_Real>& c)
{
    return c;
}


template<class Real>
inline const type::Vec<3,Real>& center(const defaulttype::RigidCoord<3,Real>& c)
{
    return c.getCenter();
}

template<class Real>
inline type::Vec<3,Real>& center(defaulttype::RigidCoord<3,Real>& c)
{
    return c.getCenter();
}

template <class DataTypes>
ShapeMatching<DataTypes>::ShapeMatching()
    : 	  iterations(initData(&iterations, (unsigned int)1, "iterations", "Number of iterations."))
    , affineRatio(initData(&affineRatio,(Real)0.0,"affineRatio","Blending between affine and rigid."))
    , fixedweight(initData(&fixedweight,(Real)1.0,"fixedweight","weight of fixed particles."))
    , fixedPosition0(initData(&fixedPosition0,"fixedPosition0","rest positions of non mechanical particles."))
    , fixedPosition(initData(&fixedPosition,"fixedPosition","current (fixed) positions of non mechanical particles."))
    , position(initData(&position,"position","Input positions."))
    , cluster(initData(&cluster,"cluster","Input clusters."))
    , targetPosition(initData(&targetPosition,"targetPosition","Computed target positions."))
    , topo(nullptr)
    , oldRestPositionSize(0)
    , oldfixedweight(0)
{
    addInput(&fixedPosition0);
    addInput(&fixedPosition);
    addInput(&position);
    addInput(&cluster);
    addOutput(&targetPosition);
}

template <class DataTypes>
void ShapeMatching<DataTypes>::init()
{
    core::behavior::SingleStateAccessor<DataTypes>::init();

    setDirtyValue();

    //- Topology Container
    this->getContext()->get(topo);

    update();
}

template <class DataTypes>
void ShapeMatching<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void ShapeMatching<DataTypes>::doUpdate()
{
    if (!this->mstate)
        return;

    const auto* restPositionsData = this->mstate->read(core::ConstVecCoordId::restPosition());
    if (!restPositionsData)
    {
        msg_error() << "Rest position cannot be found in mechanical object '" << this->mstate->getPathName() << "'";
        return;
    }

    const VecCoord& restPositions = restPositionsData->getValue();
    helper::ReadAccessor< Data< VecCoord > > fixedPositions0 = this->fixedPosition0;
    helper::ReadAccessor< Data< VecCoord > > fixedPositions = this->fixedPosition;
    helper::ReadAccessor<Data< VecCoord > > currentPositions = position;
    helper::WriteOnlyAccessor<Data< VecCoord > > targetPos = targetPosition;
    const helper::ReadAccessor<Data< VVI > > clust = cluster;

    VI::const_iterator it, itEnd;
    size_t nbp = restPositions.size() , nbf = fixedPositions0.size() , nbc = clust.size();

    msg_info() << "#clusters=" << nbc << msgendl
               << " #restpos=" << nbp << msgendl
               << " #pos=" << currentPositions.size() ;

    if(!nbc || !nbp  || !currentPositions.size()) return;

    //if mechanical state or cluster have changed, we must compute again xcm0
    if(oldRestPositionSize != nbp+nbf || oldfixedweight != this->fixedweight.getValue() || m_dataTracker.hasChanged(this->cluster))
    {
        dmsg_info() <<"shape matching: update Xcm0" ;

        T.resize(nbc);
        Qxinv.resize(nbc);
        Xcm.resize(nbc);
        Xcm0.resize(nbc);
        W.resize(nbc);
        nbClust.resize(nbp); nbClust.fill(0);
        for (size_t i=0 ; i<nbc ; ++i)
        {
            W[i] = 0;
            Xcm0[i] = Coord();
            Qxinv[i].fill(0);
            for (it = clust[i].begin(), itEnd = clust[i].end(); it != itEnd ; ++it)
            {
                Coord p0 = (*it<nbp)?restPositions[*it]:fixedPositions0[*it-nbp];
                Real w = (*it<nbp)?(Real)1.0:this->fixedweight.getValue();
                Xcm0[i] += p0*w;
                Qxinv[i] += type::dyad(p0,p0)*w;
                W[i] += w;
                if(*it<nbp) nbClust[*it]++;
            }
            Xcm0[i] /= W[i];
            Qxinv[i] -= type::dyad(Xcm0[i],Xcm0[i])*W[i]; // sum wi.(X0-Xcm0)(X0-Xcm0)^T = sum wi.X0.X0^T - W.sum(X0).Xcm0^T
            Mat3x3 inv;
            const bool canInvert = inv.invert(Qxinv[i]);
            assert(canInvert);
            SOFA_UNUSED(canInvert);
            Qxinv[i]=inv;
        }
        oldRestPositionSize = nbp+nbf;
        oldfixedweight = this->fixedweight.getValue();
    }

    targetPos.resize(nbp); 	for (size_t i=0 ; i<nbp ; ++i) targetPos[i]=currentPositions[i];

    for (unsigned int iter=0 ; iter<iterations.getValue()  ; ++iter)
    {
        // this could be speeded up using fast summation technique
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(sofa::helper::IndexOpenMP<unsigned int>::type i=0 ; i<nbc ; ++i)
        {
            Xcm[i] = Coord();
            T[i].fill(0);
            for (VI::const_iterator it = clust[i].begin() ; it != clust[i].end() ; ++it)
            {
                Coord p0 = (*it<nbp)?restPositions[*it]:fixedPositions0[*it-nbp];
                Coord p = (*it<nbp)?targetPos[*it]:fixedPositions[*it-nbp];
                Real w = (*it<nbp)?(Real)1.0:this->fixedweight.getValue();
                Xcm[i] += p*w;
                T[i] += type::dyad(p,p0)*w;
            }

            T[i] -= type::dyad(Xcm[i],Xcm0[i]); // sum wi.(X-Xcm)(X0-Xcm0)^T = sum wi.X.X0^T - sum(wi.X).Xcm0^T
            Xcm[i] /= W[i];
            Mat3x3 R;
            if(affineRatio.getValue()!=(Real)1.0)
            {
                helper::Decompose<Real>::polarDecomposition(T[i], R);
            }
            if(affineRatio.getValue()!=(Real)0.0)
                T[i] = T[i] * Qxinv[i] * (affineRatio.getValue()) + R * (1.0f-affineRatio.getValue());
            else T[i] = R;
        }

        for (size_t i=0; i<nbp; ++i) targetPos[i]=Coord();

        for (size_t i=0; i<nbc; ++i)
            for (VI::const_iterator it = clust[i].begin() ; it != clust[i].end() ; ++it)
                if(*it<nbp)
                    targetPos[*it] += Xcm[i] + T[i] *(restPositions[*it] - Xcm0[i]);

        for (size_t i=0; i<nbp; ++i)
            if(nbClust[i])
                targetPos[i] /= (Real)nbClust[i];
            else targetPos[i]=currentPositions[i];
    }
}

// Specialization for rigids
template <>
void ShapeMatching<sofa::defaulttype::Rigid3Types >::doUpdate();







template <class DataTypes>
void ShapeMatching<DataTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{

}


} //namespace sofa::component::engine::analyze
