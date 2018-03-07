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
#ifndef SOFA_COMPONENT_ENGINE_MATHOP_INL
#define SOFA_COMPONENT_ENGINE_MATHOP_INL

#include <SofaGeneralEngine/MathOp.h>

namespace sofa
{

namespace component
{

namespace engine
{

template<typename T>
struct MathOpAdd
{
    static const char* Name() { return "+"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out += in[i];
    }
};

template<typename T>
struct MathOpSub
{
    static const char* Name() { return "-"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        if (in.size() == 1)
        {
            *out = -(in[0]);
        }
        else
        {
            *out = in[0];
            for (unsigned int i=1;i<in.size();++i)
                *out -= in[i];
        }
    }
};

template<typename T>
struct MathOpMul
{
    static const char* Name() { return "*"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
        {
            *out *= in[i];
        }
    }
};

template<typename T>
struct MathOpDiv
{
    static const char* Name() { return "/"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        if (in.size() == 1)
        {
            *out = 1/(in[0]);
        }
        else
        {
            *out = in[0];
            for (unsigned int i=1;i<in.size();++i)
                *out /= in[i];
        }
    }
};

template<typename T>
struct MathOpMin
{
    static const char* Name() { return "<"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
        {
            if (in[i] < *out)
                *out = in[i];
        }
    }
};

template<typename T>
struct MathOpMax
{
    static const char* Name() { return ">"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
        {
            if (in[i] > *out)
                *out = in[i];
        }
    }
};

template<typename T>
struct MathOpNot
{
    static const char* Name() { return "!"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = !in[0];
    }
};

template<typename T>
struct MathOpAnd
{
    static const char* Name() { return "&"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out &= in[i];
    }
};

template<typename T>
struct MathOpOr
{
    static const char* Name() { return "|"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out |= in[i];
    }
};

template<typename T>
struct MathOpXor
{
    static const char* Name() { return "^"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out ^= in[i];
    }
};

template<typename T>
struct MathOpNAnd
{
    static const char* Name() { return "!&"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out &= in[i];
        *out = !out;
    }
};

template<typename T>
struct MathOpNOr
{
    static const char* Name() { return "!|"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out |= in[i];
        *out = !out;
    }
};

template<typename T>
struct MathOpXNor
{
    static const char* Name() { return "!^"; }
    void operator()(T* out, const helper::vector<T>& in)
    {
        *out = in[0];
        for (unsigned int i=1;i<in.size();++i)
            *out ^= in[i];
        *out = !out;
    }
};

template<typename T>
struct MathOpTraits
{
    typedef
        MathOpAdd <T>
        Ops;
};

/// Real-like ops
template<typename T>
struct MathOpTraitsReal
{
    typedef std::pair<std::pair<
        std::pair< MathOpAdd <T>, MathOpSub <T> > ,
        std::pair< MathOpMul <T>, MathOpDiv <T> > > ,
        std::pair< MathOpMin <T>, MathOpMax <T> >   >
        Ops;
};

template<>
class MathOpTraits<int> : public MathOpTraitsReal<int> {};

template<>
class MathOpTraits<float> : public MathOpTraitsReal<float> {};

template<>
class MathOpTraits<double> : public MathOpTraitsReal<double> {};

/// Real-like ops
template<typename T>
struct MathOpTraitsVecReal
{
    typedef 
        std::pair< MathOpAdd <T>, MathOpSub <T> >
        Ops;
};

template<int N, typename Real>
class MathOpTraits< defaulttype::Vec<N,Real> > : public MathOpTraitsVecReal< defaulttype::Vec<N,Real> > {};

/// Bool-like ops
template<typename T>
struct MathOpTraitsBool
{
    typedef std::pair<MathOpNot<T>, std::pair<std::pair<
        std::pair< MathOpOr  <T>, MathOpNOr <T> > ,
        std::pair< MathOpAnd <T>, MathOpNAnd<T> > > ,
        std::pair< MathOpXor <T>, MathOpXNor<T> > > >
        Ops;
};

template<>
class MathOpTraits<bool> : public MathOpTraitsBool<bool> {};

template<typename TOps>
struct MathOpNames
{
    static void get(helper::vector<std::string>& out)
    {
        out.push_back(TOps::Name());
    }
};

template<typename TOps1,typename TOps2>
struct MathOpNames< std::pair<TOps1, TOps2> >
{
    static void get(helper::vector<std::string>& out)
    {
        MathOpNames<TOps1>::get(out);
        MathOpNames<TOps2>::get(out);
    }
};

template<typename TOps>
struct MathOpApply
{
    static bool isSupported(const std::string& op)
    {
        return op == TOps::Name();
    }
    template<class VecValue>
    static bool apply(const std::string& op, Data<VecValue>* d_out, const helper::vector<Data<VecValue>*>& d_in)
    {
        typedef typename VecValue::value_type Value;
        if (op == TOps::Name())
        {
            TOps op;
            helper::WriteAccessor<Data<VecValue> > out = *d_out;
            helper::vector<const VecValue*> in;
            unsigned int nbin = d_in.size();
            in.reserve(nbin);
            for (unsigned int idin = 0; idin < nbin; ++idin)
            {
                in.push_back(&(d_in[idin]->getValue()));
            }
            unsigned int size = 0;
            if (nbin > 0)
            {
                size = in[0]->size();
                for (unsigned int idin = 1; idin < nbin; ++idin)
                {
                    if (in[idin]->size() < size)
                    {
                        size = in[idin]->size();
                    }
                }
            }
            out.resize(size);
            helper::vector<Value> values;
            values.resize(nbin);
            for (unsigned int idv = 0; idv < size; ++idv)
            {
                for (unsigned int idin = 0; idin < nbin; ++idin)
                {
                    values[idin] = (*in[idin])[idv];
                }
                Value o;
                op(&o, values);
                out[idv] = o;
            }
            return true;
        }
        else
        {
            return false;
        }
    }
};

template<typename TOps1, typename TOps2>
struct MathOpApply< std::pair<TOps1, TOps2> >
{
    static bool isSupported(const std::string& op)
    {
        return  MathOpApply<TOps1>::isSupported(op) ||  MathOpApply<TOps2>::isSupported(op);
    }
    template<class VecValue>
    static bool apply(const std::string& op, Data<VecValue>* d_out, const helper::vector<Data<VecValue>*>& d_in)
    {
        return  MathOpApply<TOps1>::apply(op, d_out, d_in) ||  MathOpApply<TOps2>::apply(op, d_out, d_in);
    }
};

template <class VecT>
MathOp<VecT>::MathOp()
    : f_nbInputs (initData(&f_nbInputs, (unsigned)2, "nbInputs", "Number of input values"))
    , f_op       (initData(&f_op      ,              "op"      , "Selected operation to apply"))
    , f_output   (initData(&f_output  ,              "output"  , "Output values"))
{
    
    sofa::helper::OptionsGroup& ops = *f_op.beginEdit();
    helper::vector<std::string> opnames;
    MathOpNames< typename MathOpTraits<Value>::Ops >::get(opnames);
    ops.setNbItems(opnames.size());
    for (unsigned int i = 0; i < opnames.size(); ++i)
        ops.setItemName(i, opnames[i]);
    //ops.setSelectedItem(0);
    f_op.endEdit();
    createInputs();
}

template <class VecT>
MathOp<VecT>::~MathOp()
{
    for (unsigned int i=0; i<vf_inputs.size(); ++i)
    {
        this->delInput(vf_inputs[i]);
        delete vf_inputs[i];
    }
    vf_inputs.clear();
}

template <class VecT>
void MathOp<VecT>::createInputs(int nb)
{
    unsigned int n = (nb < 0) ? f_nbInputs.getValue() : (unsigned int)nb;
    for (unsigned int i=vf_inputs.size(); i<n; ++i)
    {
        std::ostringstream oname, ohelp;
        oname << "input" << (i+1);
        ohelp << "input values " << (i+1);
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
        f_nbInputs.setValue(n);
}

/// Parse the given description to assign values to this object's fields and potentially other parameters
template <class VecT>
void MathOp<VecT>::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
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
void MathOp<VecT>::parseFields ( const std::map<std::string,std::string*>& str )
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
void MathOp<VecT>::init()
{
    addInput(&f_nbInputs);
    addInput(&f_op);
    createInputs();
    addOutput(&f_output);
    
    std::string op = f_op.getValue().getSelectedItem();
    bool result = MathOpApply< typename MathOpTraits<Value>::Ops >::isSupported(op);
    if (!result)
        serr << "Operation " << op << " NOT SUPPORTED" << sendl;

    setDirtyValue();
}

template <class VecT>
void MathOp<VecT>::reinit()
{
    createInputs();

    update();
}

template <class VecT>
void MathOp<VecT>::update()
{
//    createInputs();
    std::string op = f_op.getValue().getSelectedItem();

    // ensure all inputs are up-to-date before cleaning engine
    for (unsigned int i=0, iend=vf_inputs.size(); i<iend; ++i)
        vf_inputs[i]->updateIfDirty();

    cleanDirty();

    bool result = MathOpApply< typename MathOpTraits<Value>::Ops >::apply(
        op, &f_output, vf_inputs);
    if (!result)
        serr << "Operation " << op << " FAILED" << sendl;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
