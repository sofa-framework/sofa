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
#ifndef SOFA_COMPONENT_ENGINE_SHAPEMATCHING_INL
#define SOFA_COMPONENT_ENGINE_SHAPEMATCHING_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/ShapeMatching.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace engine
{
template<class Real>
inline const Vec<3,Real>& center(const Vec<3,Real>& c)
{
    return c;
}

template<class _Real>
inline Vec<3,_Real>& center(Vec<3,_Real>& c)
{
    return c;
}


inline const Vec<3,double>& center(const StdRigidTypes<3,double>::Coord& c)
{
    return c.getCenter();
}

inline Vec<3,double>& center(StdRigidTypes<3,double>::Coord& c)
{
    return c.getCenter();
}

inline const Vec<3,float>& center(const StdRigidTypes<3,float>::Coord& c)
{
    return c.getCenter();
}

inline Vec<3,float>& center(StdRigidTypes<3,float>::Coord& c)
{
    return c.getCenter();
}


using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;

template <class DataTypes>
ShapeMatching<DataTypes>::ShapeMatching()
    : 	  iterations(initData(&iterations, (unsigned int)1, "iterations", "Number of iterations."))
    , affineRatio(initData(&affineRatio,(Real)0.0,"affineRatio","Blending between affine and rigid."))
    , position(initData(&position,"position","Input positions."))
    , cluster(initData(&cluster,"cluster","Input clusters."))
    , targetPosition(initData(&targetPosition,"targetPosition","Computed target positions."))
    , topo(NULL)
{
    //affineRatio.setWidget("0to1RatioWidget");
}

template <class DataTypes>
void ShapeMatching<DataTypes>::init()
{
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    addInput(&position);
    addInput(&cluster);
    addOutput(&targetPosition);
    setDirtyValue();

    //- Topology Container
    this->getContext()->get(topo);

    oldRestPositionSize = oldClusterSize = oldClusterSize0 = 0;

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
    cleanDirty();

    const VecCoord& restPositions = *mstate->getX0();
    helper::ReadAccessor<Data< VecCoord > > currentPositions = position;
    helper::WriteAccessor<Data< VecCoord > > targetPos = targetPosition;
    helper::ReadAccessor<Data< VVI > > clust = cluster;

    //this->mstate->resize(restPositions.size());

    VI::const_iterator it, itEnd;
    unsigned int nbp = restPositions.size() , nbc = clust.size();

    if (this->f_printLog.getValue())
    {
        std::cout<<"ShapeMatching: #clusters="<<nbc<<std::endl;
        std::cout<<"ShapeMatching: #restpos="<<nbp<<std::endl;
        std::cout<<"ShapeMatching: #pos="<<currentPositions.size()<<std::endl;
    }

    if(!nbc || !nbp  || !currentPositions.size()) return;

    //if mechanical state or cluster have changed, we must compute again xcm0
    if(oldRestPositionSize != nbp || oldClusterSize0 != clust[0].size() || oldClusterSize != nbc)
    {
        T.resize(nbc);
        Qxinv.resize(nbc);
        Xcm.resize(nbc);
        Xcm0.resize(nbc);
        nbClust.resize(nbp); nbClust.fill(0);
        for (unsigned int i=0 ; i<nbc ; ++i)
        {
            Xcm0[i] = Coord();
            Qxinv[i].fill(0);
            for (it = clust[i].begin(), itEnd = clust[i].end(); it != itEnd ; ++it)
            {
                Xcm0[i] += restPositions[*it];
                Qxinv[i] += covNN(restPositions[*it],restPositions[*it]);
                nbClust[*it]++;
            }
            Xcm0[i] /= clust[i].size();
            Qxinv[i] -= covNN(Xcm0[i],Xcm0[i])*clust[i].size(); // sum(X0-Xcm0)(X0-Xcm0)^T = sum X0.X0^T - sum(X0).Xcm0^T
            Mat3x3 inv; inv.invert(Qxinv[i]);  Qxinv[i]=inv;
        }

        oldRestPositionSize = restPositions.size();
        oldClusterSize0 = clust[0].size();
        oldClusterSize = nbc;
    }

    targetPos.resize(nbp); 	for (unsigned int i=0 ; i<nbp ; ++i) targetPos[i]=currentPositions[i];

    for (unsigned int iter=0 ; iter<iterations.getValue()  ; ++iter)
    {
        // this could be parallelize or speed up using fast summation technique
        for (unsigned int i=0 ; i<nbc ; ++i)
        {
            Xcm[i] = Coord();
            T[i].fill(0);
            for (it = clust[i].begin(), itEnd = clust[i].end(); it != itEnd ; ++it)
            {
                Xcm[i] += targetPos[*it];
                T[i] += covNN(targetPos[*it],restPositions[*it]);
            }
        }


        for (unsigned int i=0 ; i<nbc ; ++i)
        {
            T[i] -= covNN(Xcm[i],Xcm0[i]); // sum(X-Xcm)(X0-Xcm0)^T = sum X.X0^T - sum(X).Xcm0^T
            Xcm[i] /= clust[i].size();
            Mat3x3 R,S;
            if(affineRatio.getValue()!=(Real)1.0)
            {
                polar_decomp(T[i], R, S);
                //if (determinant(R) < 0) for(unsigned int j=0 ; j<3;j++) R[j][0] *= -1;  // handle symmetry
            }
            if(affineRatio.getValue()!=(Real)0.0)
                T[i] = T[i] * Qxinv[i] * (affineRatio.getValue()) + R * (1.0-affineRatio.getValue());
            else T[i] = R;
        }

        for (unsigned int i=0; i<nbp; ++i)
            targetPos[i]=Coord();

        for (unsigned int i=0; i<nbc; ++i)
            for (it = clust[i].begin(), itEnd = clust[i].end(); it != itEnd ; ++it)
                targetPos[*it] += Xcm[i] + T[i] *(restPositions[*it] - Xcm0[i]);

        for (unsigned int i=0; i<nbp; ++i)
            if(nbClust[i])
                targetPos[i] /= (Real)nbClust[i];
            else targetPos[i]=currentPositions[i];
    }

}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void ShapeMatching<Rigid3dTypes >::update();
#endif
#ifndef SOFA_DOUBLE
template <>
void ShapeMatching<Rigid3fTypes >::update();
#endif






template <class DataTypes>
void ShapeMatching<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowForceFields())
    {
        /*  const VecCoord& currentPositions = *mstate->getX();

          glPushAttrib( GL_LIGHTING_BIT);

          glDisable(GL_LIGHTING);

          glBegin(GL_LINES);

          glEnd();

        glPopAttrib(); */
    }
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
