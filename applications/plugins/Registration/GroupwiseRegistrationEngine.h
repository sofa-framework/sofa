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
#ifndef SOFA_GroupwiseRegistrationEngine_H
#define SOFA_GroupwiseRegistrationEngine_H

#include <Registration/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/decompose.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/vectorData.h>

namespace sofa
{

namespace component
{

namespace engine
{


/**
 * Register a set of meshes of similar topology
 */


template <class T>
class GroupwiseRegistrationEngine : public core::DataEngine
{

public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(GroupwiseRegistrationEngine,T),Inherited);

    typedef typename T::Coord Coord;
    typedef typename T::VecCoord VecCoord;
    typedef typename T::Real Real;
    enum {sizeT = T::deriv_total_size };

    typedef defaulttype::Mat<sizeT,sizeT,Real> affine;

    Data<unsigned int> f_nbInputs; ///< Number of input vectors
    helper::vectorData<VecCoord> vf_inputs;
    helper::vectorData<VecCoord> vf_outputs;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const GroupwiseRegistrationEngine<T>* = NULL) { return T::Name();   }

    virtual void init()
    {
        addInput(&f_nbInputs);
        vf_inputs.resize(f_nbInputs.getValue());
        vf_outputs.resize(f_nbInputs.getValue());
        setDirtyValue();
    }

    virtual void reinit()
    {
        vf_inputs.resize(f_nbInputs.getValue());
        vf_outputs.resize(f_nbInputs.getValue());
        update();
    }

protected:

    GroupwiseRegistrationEngine()    :   Inherited()
      , f_nbInputs (initData(&f_nbInputs, (unsigned)2, "nbInputs", "Number of input vectors"))
      , vf_inputs(this, "input", "input vector ", helper::DataEngineInput)
      , vf_outputs(this, "output", "output vector", helper::DataEngineOutput)
    {
        vf_inputs.resize(f_nbInputs.getValue());
        vf_outputs.resize(f_nbInputs.getValue());
    }


    virtual ~GroupwiseRegistrationEngine()
    {
    }

    virtual void update()
    {
        updateAllInputsIfDirty();
        cleanDirty();

        const unsigned int M = vf_inputs.size();
        if(!M) return;

        helper::ReadAccessor<Data<VecCoord> > pos0(vf_inputs[0]);
        helper::WriteOnlyAccessor<Data<VecCoord> > outPos0(vf_outputs[0]);

        const unsigned int N = pos0.size();

        // register all points to the first set
        outPos0.resize(N);
        for (unsigned int i=0; i<N; ++i) outPos0[i]=pos0[i]; // copy first

        for (unsigned int i=1; i<M; ++i)
        {
            helper::ReadAccessor<Data<VecCoord> > pos(vf_inputs[i]);
            helper::WriteOnlyAccessor<Data<VecCoord> > outPos(vf_outputs[i]);
            if(N!=pos.size()) { serr<<"input"<<i+1<<" has an invalid size"<<sendl; return; }

            affine R; Coord t;
            ClosestRigid(pos.ref(), pos0.ref(), R, t);

            outPos.resize(N);
            for(unsigned int j=0; j<N; ++j) outPos[j] = R*pos[j] + t;
        }
    }


    void ClosestRigid(const VecCoord& source, const VecCoord& target, affine& R, Coord& t)
    {
        unsigned int N=source.size();

        Coord Xcm0,Xcm;
        affine M;
        M.fill(0);

        for(unsigned int i=0;i<N;i++)
        {
            Xcm+=target[i];
            Xcm0+=source[i];
            M += dyad(target[i],source[i]);
        }
        Xcm /= (Real)N;
        M -= dyad(Xcm,Xcm0); // sum (X-Xcm)(X0-Xcm0)^T = sum X.X0^T - N.Xcm.Xcm0^T
        Xcm0 /= (Real)N;
        helper::Decompose<Real>::polarDecomposition(M, R);
        t = Xcm - R*Xcm0;
    }


public:

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        vf_inputs.parseSizeData(arg, f_nbInputs);
        vf_outputs.parseSizeData(arg, f_nbInputs);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        vf_inputs.parseFieldsSizeData(str, f_nbInputs);
        vf_outputs.parseFieldsSizeData(str, f_nbInputs);
        Inherit1::parseFields(str);
    }


};


} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_GroupwiseRegistrationEngine_H
