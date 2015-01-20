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
#ifndef SOFA_GroupwiseRegistrationEngine_H
#define SOFA_GroupwiseRegistrationEngine_H

#include "initRegistration.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/decompose.h>

#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace component
{

namespace engine
{

using helper::vector;
using defaulttype::Vec;
using defaulttype::Mat;

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

    Data<unsigned int> f_nbInputs;
    helper::vector<Data<VecCoord>*> vf_inputs;
    helper::vector<Data<VecCoord>*> vf_outputs;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const GroupwiseRegistrationEngine<T>* = NULL) { return T::Name();   }

    virtual void init()
    {
        addInput(&f_nbInputs);
        createDataVectors();
        setDirtyValue();
    }

    virtual void reinit()
    {
        createDataVectors();
        update();
    }

protected:

    GroupwiseRegistrationEngine()    :   Inherited()
      , f_nbInputs (initData(&f_nbInputs, (unsigned)2, "nbInputs", "Number of input vectors"))
    {
        createDataVectors();
    }

    virtual ~GroupwiseRegistrationEngine()
    {
        deleteDataVectors();
    }

    virtual void update()
    {
        cleanDirty();
        createDataVectors();

        const unsigned int M = vf_inputs.size();
        if(!M) return;

        helper::ReadAccessor<Data<VecCoord> > pos0(vf_inputs[0]);
        helper::WriteAccessor<Data<VecCoord> > outPos0(vf_outputs[0]);

        const unsigned int N = pos0.size();

        // register all points to the first set
        outPos0.resize(N);
        for (unsigned int i=0; i<N; ++i) outPos0[i]=pos0[i]; // copy first

        for (unsigned int i=1; i<M; ++i)
        {
            helper::ReadAccessor<Data<VecCoord> > pos(vf_inputs[i]);
            helper::WriteAccessor<Data<VecCoord> > outPos(vf_outputs[i]);
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


    void createDataVectors(int nb=-1)
    {
        unsigned int n = (nb < 0) ? f_nbInputs.getValue() : (unsigned int)nb;
        for (unsigned int i=vf_inputs.size(); i<n; ++i)
        {
            std::ostringstream oname, ohelp;
            oname << "input" << (i+1);
            ohelp << "input vector " << (i+1);
            std::string name_i = oname.str();
            std::string help_i = ohelp.str();
            Data<VecCoord>* d = new Data<VecCoord>(help_i.c_str(), true, false);
            d->setName(name_i);
            vf_inputs.push_back(d);
            this->addData(d);
            this->addInput(d);
        }
        for (unsigned int i = n; i < vf_inputs.size(); ++i)
        {
            this->delInput(vf_inputs[i]);
            delete vf_inputs[i];
        }
        vf_inputs.resize(n);
        if (n != f_nbInputs.getValue())
            f_nbInputs.setValue(n);



        for (unsigned int i=vf_outputs.size(); i<n; ++i)
        {
            std::ostringstream oname, ohelp;
            oname << "output" << (i+1);
            ohelp << "output vector " << (i+1);
            std::string name_i = oname.str();
            std::string help_i = ohelp.str();
            Data<VecCoord>* d = new Data<VecCoord>(help_i.c_str(), true, false);
            d->setName(name_i);
            vf_outputs.push_back(d);
            this->addData(d);
            this->addOutput(d);
        }
        for (unsigned int i = n; i < vf_outputs.size(); ++i)
        {
            this->delOutput(vf_outputs[i]);
            delete vf_outputs[i];
        }
        vf_outputs.resize(n);
    }

    void deleteDataVectors()
    {
        for (unsigned int i=0; i<vf_inputs.size(); ++i)
        {
            this->delInput(vf_inputs[i]);
            delete vf_inputs[i];
        }
        vf_inputs.clear();

        for (unsigned int i=0; i<vf_outputs.size(); ++i)
        {
            this->delOutput(vf_outputs[i]);
            delete vf_outputs[i];
        }
        vf_outputs.clear();
    }

public:

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        const char* p = arg->getAttribute(f_nbInputs.getName().c_str());
        if (p)
        {
            std::string nbStr = p;
            sout << "parse: setting nbInputs="<<nbStr<<sendl;
            f_nbInputs.read(nbStr);
            createDataVectors();
        }
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(f_nbInputs.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            sout << "parseFields: setting nbInputs="<<nbStr<<sendl;
            f_nbInputs.read(nbStr);
            createDataVectors();
        }
        Inherit1::parseFields(str);
    }


};


} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_GroupwiseRegistrationEngine_H
