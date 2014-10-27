/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MERGEVECTORS_INL
#define SOFA_COMPONENT_ENGINE_MERGEVECTORS_INL

#include <SofaEngine/MergeVectors.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class VecT>
MergeVectors<VecT>::MergeVectors()
    : f_nbInputs (initData(&f_nbInputs, (unsigned)2, "nbInputs", "Number of input vectors"))
    , f_output   (initData(&f_output  ,              "output"  , "Output vector"))
{
    createInputs();
}

template <class VecT>
MergeVectors<VecT>::~MergeVectors()
{
    for (unsigned int i=0; i<vf_inputs.size(); ++i)
    {
        this->delInput(vf_inputs[i]);
        delete vf_inputs[i];
    }
    vf_inputs.clear();
}

template <class VecT>
void MergeVectors<VecT>::createInputs(int nb)
{
    unsigned int n = (nb < 0) ? f_nbInputs.getValue() : (unsigned int)nb;
    for (unsigned int i=vf_inputs.size(); i<n; ++i)
    {
        std::ostringstream oname, ohelp;
        oname << "input" << (i+1);
        ohelp << "input vector " << (i+1);
        std::string name_i = oname.str();
        std::string help_i = ohelp.str();
        Data<VecValue>* d = new Data<VecValue>(help_i.c_str(), true, false);
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
        f_nbInputs.setValue(n,true);
}

/// Parse the given description to assign values to this object's fields and potentially other parameters
template <class VecT>
void MergeVectors<VecT>::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    const char* p = arg->getAttribute(f_nbInputs.getName().c_str());
    if (p)
    {
        std::string nbStr = p;
        sout << "parse: setting nbInputs="<<nbStr<<sendl;
        f_nbInputs.read(nbStr);
        createInputs();
    }
    Inherit1::parse(arg);
}

/// Assign the field values stored in the given map of name -> value pairs
template <class VecT>
void MergeVectors<VecT>::parseFields ( const std::map<std::string,std::string*>& str )
{
    std::map<std::string,std::string*>::const_iterator it = str.find(f_nbInputs.getName());
    if (it != str.end() && it->second)
    {
        std::string nbStr = *it->second;
        sout << "parseFields: setting nbInputs="<<nbStr<<sendl;
        f_nbInputs.read(nbStr);
        createInputs();
    }
    Inherit1::parseFields(str);
}

template <class VecT>
void MergeVectors<VecT>::init()
{
    addInput(&f_nbInputs);
    createInputs();
    addOutput(&f_output);
    setDirtyValue();
}

template <class VecT>
void MergeVectors<VecT>::reinit()
{
    createInputs();
    update();
}

template <class VecT>
void MergeVectors<VecT>::update()
{
    cleanDirty();
    createInputs();

    VecValue* out = f_output.beginEdit();
    helper::vector<const VecValue*> in;
    unsigned int nbin = vf_inputs.size();
    in.reserve(nbin);
    for (unsigned int idin = 0; idin < nbin; ++idin)
    {
        in.push_back(&(vf_inputs[idin]->getValue()));
    }
    unsigned int size = 0;
    if (nbin > 0)
    {
        for (unsigned int idin = 0; idin < nbin; ++idin)
        {
            size += in[idin]->size();
        }
    }
    out->clear();
    out->reserve(size);
    for (unsigned int idin = 0; idin < nbin; ++idin)
    {
        out->insert(out->end(),in[idin]->begin(), in[idin]->end());
    }
    f_output.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
