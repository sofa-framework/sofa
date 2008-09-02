/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <vector>
#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#endif
namespace sofa
{

namespace component
{

namespace mapping
{

typedef enum
{
    WEIGHT_LINEAR, WEIGHT_INVDIST, WEIGHT_HERMITE
} WeightingType;

typedef enum
{
    INTERPOLATION_LINEAR, INTERPOLATION_DUAL_QUATERNION
} InterpolationType;

template <class BasicMapping>
class SkinningMapping : public BasicMapping, public virtual core::objectmodel::BaseObject
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::SparseDeriv InSparseDeriv;
    typedef typename Coord::value_type Real;
    enum { N=Coord::static_size };
    typedef defaulttype::Mat<N,N,Real> Mat;

#ifdef SOFA_DEV
    typedef typename helper::DualQuatd DualQuat;
#endif
protected:
    sofa::helper::vector<InCoord> initPosDOFs; // translation and rotation of the blended reference frame i, where i=0..n.
    sofa::helper::vector<Coord> initPos; // pos: point coord in  the reference frame i, where i=0..n ( + 1 for the blended reference frame)
    sofa::helper::vector<Coord> initBlendedPos; // pos: point coord in  the blended reference frames, where i=0..n ( + 1 for the blended reference frame)
    sofa::helper::vector<Coord> rotatedPoints;

    Data<sofa::helper::vector<unsigned int> > repartition;
    Data<sofa::helper::vector<double> >  coefs;
    Data<unsigned int> nbRefs;

    bool computeWeights;
    WeightingType wheighting;
    InterpolationType interpolation;

    class Loader;
    void load ( const char* filename );
    inline void computeInitPos();
    inline void sortReferences();

public:
    SkinningMapping ( In* from, Out* to );
    virtual ~SkinningMapping();

    void init();
    void parse ( core::objectmodel::BaseObjectDescription* arg );

    void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT ( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();
    void clear();

    // Weights
    void setWeightsToHermite();
    void setWieghtsToInvDist();
    void setWeightsToLinear();
    inline void updateWeights();

    // Interpolations
    void setInterpolationToLinear();
    void setInterpolationToDualQuaternion();

    // Accessors
    void setNbRefs ( unsigned int nb ) { nbRefs.setValue ( nb ); }
    void setWeightCoefs ( sofa::helper::vector<double> &weights );
    void setRepartition ( sofa::helper::vector<unsigned int> &rep );
    void setComputeWeights ( bool val ) {computeWeights=val;}
    unsigned int getNbRefs() { return nbRefs.getValue(); }
    const sofa::helper::vector<double>& getWeightCoefs() { return coefs.getValue(); }
    const sofa::helper::vector<unsigned int>& getRepartition() { return repartition.getValue(); }
    bool getComputeWeights() { return computeWeights; }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
