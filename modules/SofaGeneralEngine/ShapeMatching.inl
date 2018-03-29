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
#ifndef SOFA_COMPONENT_ENGINE_SHAPEMATCHING_INL
#define SOFA_COMPONENT_ENGINE_SHAPEMATCHING_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/visual/VisualParams.h>
#include <SofaGeneralEngine/ShapeMatching.h>
#include <sofa/helper/decompose.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <sofa/helper/IndexOpenMP.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace sofa
{

namespace component
{

namespace engine
{
template<class Real>
inline const defaulttype::Vec<3,Real>& center(const defaulttype::Vec<3,Real>& c)
{
    return c;
}

template<class _Real>
inline defaulttype::Vec<3,_Real>& center(defaulttype::Vec<3,_Real>& c)
{
    return c;
}


template<class Real>
inline const defaulttype::Vec<3,Real>& center(const defaulttype::RigidCoord<3,Real>& c)
{
    return c.getCenter();
}

template<class Real>
inline defaulttype::Vec<3,Real>& center(defaulttype::RigidCoord<3,Real>& c)
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
    , topo(NULL)
    , oldRestPositionSize(0)
    , oldfixedweight(0)
{
    //affineRatio.setWidget("0to1RatioWidget");
}

template <class DataTypes>
void ShapeMatching<DataTypes>::init()
{
    mstate = dynamic_cast< sofa::core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    addInput(&fixedPosition0);
    addInput(&fixedPosition);
    addInput(&position);
    addInput(&cluster);
    addOutput(&targetPosition);
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
void ShapeMatching<DataTypes>::update()
{
    bool clusterdirty = this->cluster.isDirty();

    const VecCoord& restPositions = mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    helper::ReadAccessor< Data< VecCoord > > fixedPositions0 = this->fixedPosition0;
    helper::ReadAccessor< Data< VecCoord > > fixedPositions = this->fixedPosition;
    helper::ReadAccessor<Data< VecCoord > > currentPositions = position;
    helper::WriteOnlyAccessor<Data< VecCoord > > targetPos = targetPosition;
    helper::ReadAccessor<Data< VVI > > clust = cluster;

    //this->mstate->resize(restPositions.size());

    VI::const_iterator it, itEnd;
    size_t nbp = restPositions.size() , nbf = fixedPositions0.size() , nbc = clust.size();

    msg_info() << "#clusters=" << nbc << msgendl
               << " #restpos=" << nbp << msgendl
               << " #pos=" << currentPositions.size() ;

    if(!nbc || !nbp  || !currentPositions.size()) return;

    //if mechanical state or cluster have changed, we must compute again xcm0
    if(oldRestPositionSize != nbp+nbf || oldfixedweight != this->fixedweight.getValue() || clusterdirty)
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
                Qxinv[i] += dyad(p0,p0)*w;
                W[i] += w;
                if(*it<nbp) nbClust[*it]++;
            }
            Xcm0[i] /= W[i];
            Qxinv[i] -= dyad(Xcm0[i],Xcm0[i])*W[i]; // sum wi.(X0-Xcm0)(X0-Xcm0)^T = sum wi.X0.X0^T - W.sum(X0).Xcm0^T
            Mat3x3 inv; inv.invert(Qxinv[i]);  Qxinv[i]=inv;
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
                T[i] += dyad(p,p0)*w;
            }

            T[i] -= dyad(Xcm[i],Xcm0[i]); // sum wi.(X-Xcm)(X0-Xcm0)^T = sum wi.X.X0^T - sum(wi.X).Xcm0^T
            Xcm[i] /= W[i];
            Mat3x3 R;
            if(affineRatio.getValue()!=(Real)1.0)
            {
                helper::Decompose<Real>::polarDecomposition(T[i], R);
                //if (determinant(R) < 0) for(unsigned int j=0 ; j<3;j++) R[j][0] *= -1;  // handle symmetry
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

    cleanDirty();
}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void ShapeMatching<sofa::defaulttype::Rigid3dTypes >::update();
#endif
#ifndef SOFA_DOUBLE
template <>
void ShapeMatching<sofa::defaulttype::Rigid3fTypes >::update();
#endif






template <class DataTypes>
void ShapeMatching<DataTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{

}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
