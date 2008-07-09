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

namespace sofa
{

namespace component
{

namespace mapping
{

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

protected:
    sofa::helper::vector<InCoord> initPos;
    Coord translation;
    Mat rotation;
    sofa::helper::vector<Coord> rotatedPoints;

    class Loader;
    void load(const char* filename);

    Data<sofa::helper::vector<unsigned int> > repartition;
    Data<sofa::helper::vector<double> >  coefs;
    Data<unsigned int> nbRefs;

    bool computeWeights;

public:

    SkinningMapping(In* from, Out* to)
        : Inherit(from, to)
        , repartition(initData(&repartition,"repartition","repartition between input DOFs and skinned vertices"))
        , coefs(initData(&coefs,"coefs","weights list for the influences of the references Dofs"))
        , nbRefs(initData(&nbRefs,(unsigned)3,"nbRefs","nb references for skinning"))
        , computeWeights(true)
    {
    }

    virtual ~SkinningMapping() {}

    void init();

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("filename"))
            this->load(arg->getAttribute("filename"));
        this->Inherit::parse(arg);
    }

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );


    void draw();

    void clear();

    void setNbRefs(unsigned int nb) { nbRefs.setValue(nb); }
    void setWeightCoefs(sofa::helper::vector<double> &weights);
    void setRepartition(sofa::helper::vector<unsigned int> &rep);
    void setComputeWeights(bool val) {computeWeights=val;}

    unsigned int getNbRefs() { return nbRefs.getValue(); }
    const sofa::helper::vector<double>& getWeightCoefs() { return coefs.getValue(); }
    const sofa::helper::vector<unsigned int>& getRepartition() { return repartition.getValue(); }
    bool getComputeWeights() { return computeWeights; }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
