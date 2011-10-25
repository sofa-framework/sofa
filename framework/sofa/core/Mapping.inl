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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_MAPPING_INL
#define SOFA_CORE_MAPPING_INL

#include <sofa/core/Mapping.h>
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif
#include <iostream>

namespace sofa
{

namespace core
{

template <class In, class Out>
Mapping<In,Out>::Mapping(State<In>* from, State<Out>* to)
    : BaseMapping()
    , fromModel(from), toModel(to)
    , m_inputObject(initData(&m_inputObject, "input", "Input object to map"))
    , m_outputObject(initData(&m_outputObject, "output", "Output object to map"))
    , f_applyRestPosition( initData( &f_applyRestPosition, false, "applyRestPosition", "set to true to apply this mapping to restPosition at init"))
    , f_checkJacobian( initData( &f_checkJacobian, false, "checkJacobian", "set to true to compare results of applyJ/applyJT methods with multiplication with the matrix given by getJ()" ) )
{
    if(to != NULL && !testMechanicalState(to))
        setNonMechanical();
}

template <class In, class Out>
Mapping<In,Out>::~Mapping()
{
}

template <class In, class Out>
State<In>* Mapping<In,Out>::getFromModel()
{
    return this->fromModel;
}

template <class In, class Out>
State<Out>* Mapping<In,Out>::getToModel()
{
    return this->toModel;
}

template <class In, class Out>
helper::vector<BaseState*> Mapping<In,Out>::getFrom()
{
    helper::vector<BaseState*> vec(1,this->fromModel);
    return  vec;
}

template <class In, class Out>
helper::vector<BaseState*> Mapping<In,Out>::getTo()
{
    helper::vector<BaseState*> vec(1,this->toModel);
    return vec;
}

///<TO REMOVE>
///Necessary ?
template <class In, class Out>
helper::vector<behavior::BaseMechanicalState*> Mapping<In,Out>::getMechFrom()
{
    helper::vector<behavior::BaseMechanicalState*> vec;
    behavior::BaseMechanicalState* meshFrom = dynamic_cast<behavior::BaseMechanicalState*> (this->fromModel);
    if(meshFrom)
        vec.push_back(meshFrom);

    return vec;
}

template <class In, class Out>
helper::vector<behavior::BaseMechanicalState*> Mapping<In,Out>::getMechTo()
{
    helper::vector<behavior::BaseMechanicalState*> vec;
    behavior::BaseMechanicalState* meshTo = dynamic_cast<behavior::BaseMechanicalState*> (this->toModel);
    if(meshTo)
        vec.push_back(meshTo);

    return vec;
}

template <class In, class Out>
void Mapping<In,Out>::init()
{
    this->m_inputObject.setReadOnly(true);
    this->m_outputObject.setReadOnly(true);

    if(toModel != NULL && !testMechanicalState(toModel))
        setNonMechanical();

    apply(MechanicalParams::defaultInstance() /* PARAMS FIRST */, VecCoordId::position(), ConstVecCoordId::position());
    applyJ(MechanicalParams::defaultInstance() /* PARAMS FIRST */, VecDerivId::velocity(), ConstVecDerivId::velocity());
    if (f_applyRestPosition.getValue())
        apply(MechanicalParams::defaultInstance() /* PARAMS FIRST */, VecCoordId::restPosition(), ConstVecCoordId::restPosition());
}

template <class In, class Out>
sofa::defaulttype::BaseMatrix* Mapping<In,Out>::createMappedMatrix(const behavior::BaseMechanicalState* state1, const behavior::BaseMechanicalState* state2, func_createMappedMatrix m_createMappedMatrix)
{
    sofa::defaulttype::BaseMatrix* result;
    if( !this->areMatricesMapped() )
    {
        sout << "Mapping::createMappedMatrix() this mapping do not support matrices building. Set mapMatrices to true" << getClassName() << sendl;
        return NULL;
    }

    result = (*m_createMappedMatrix)(state1,state2);

    return result;

}




#ifdef SOFA_SMP
template<class T>
struct ParallelMappingApply
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, void *m, Shared_rw< objectmodel::Data< typename T::Out::VecCoord > > out, Shared_r< objectmodel::Data< typename T::In::VecCoord > > in)
    {
        ((T *)m)->apply(mparams /* PARAMS FIRST */, out.access(), in.read());
    }
};

template<class T>
struct ParallelMappingApplyJ
{
    void operator()(const MechanicalParams* mparams /* PARAMS FIRST */, void *m, Shared_rw< objectmodel::Data< typename T::Out::VecDeriv> > out, Shared_r< objectmodel::Data< typename T::In::VecDeriv> > in)
    {
        ((T *)m)->applyJ(mparams /* PARAMS FIRST */, out.access(), in.read());
    }
};
#endif /* SOFA_SMP */

template <class In, class Out>
void Mapping<In,Out>::apply(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecCoordId outPos, ConstMultiVecCoordId inPos)
{
    if(this->fromModel && this->toModel)
    {
        OutDataVecCoord* out = outPos[toModel].write();
        const InDataVecCoord* in = inPos[fromModel].read();
        if(out && in)
        {
#ifdef SOFA_SMP
            if (mparams->execMode() == ExecParams::EXEC_KAAPI)
                Task<ParallelMappingApply< Mapping<In,Out> > >(mparams /* PARAMS FIRST */, this,
                        **defaulttype::getShared(*out), **defaulttype::getShared(*in));
            else
#endif /* SOFA_SMP */
                this->apply(mparams /* PARAMS FIRST */, *out, *in);
        }
    }
}// Mapping::apply

template <class In, class Out>
void Mapping<In,Out>::applyJ(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId outVel, ConstMultiVecDerivId inVel)
{
    if(this->fromModel && this->toModel)
    {
        OutDataVecDeriv* out = outVel[toModel].write();
        const InDataVecDeriv* in = inVel[fromModel].read();
        if(out && in)
        {
            if (this->isMechanical() && this->f_checkJacobian.getValue(mparams))
            {
                checkApplyJ(*out->beginEdit(mparams), in->getValue(mparams), this->getJ(mparams));
                out->endEdit(mparams);
            }
            else
            {
#ifdef SOFA_SMP
                if (mparams->execMode() == ExecParams::EXEC_KAAPI)
                    Task<ParallelMappingApplyJ< Mapping<In,Out> > >(mparams /* PARAMS FIRST */, this,
                            **defaulttype::getShared(*out), **defaulttype::getShared(*in));
                else
#endif /* SOFA_SMP */
                    this->applyJ(mparams /* PARAMS FIRST */, *out, *in);
            }
        }
    }
}// Mapping::applyJ

template <class In, class Out>
void Mapping<In,Out>::applyJT(const MechanicalParams *mparams /* PARAMS FIRST */, MultiVecDerivId inForce, ConstMultiVecDerivId outForce)
{
    if(this->fromModel && this->toModel)
    {
        InDataVecDeriv* out = inForce[fromModel].write();
        const OutDataVecDeriv* in = outForce[toModel].read();
        if(out && in)
        {
            if (this->isMechanical() && this->f_checkJacobian.getValue(mparams))
            {
                checkApplyJT(*out->beginEdit(mparams), in->getValue(mparams), this->getJ(mparams));
                out->endEdit(mparams);
            }
            else
                this->applyJT(mparams /* PARAMS FIRST */, *out, *in);
        }
    }
}// Mapping::applyJT

template <class In, class Out>
void Mapping<In,Out>::disable()
{
}

template <class In, class Out>
void Mapping<In,Out>::setModels(State<In>* from, State<Out>* to)
{
    this->fromModel = from;
    this->toModel = to;
    if(to != NULL && !testMechanicalState(to))
        setNonMechanical();
}

template <class In, class Out>
std::string Mapping<In,Out>::templateName(const Mapping<In, Out>* /*mapping*/)
{
//	return std::string("Mapping<") + In::Name() + std::string(",") + Out::Name() + std::string(">");
    return In::Name() + std::string(",") + Out::Name();
}


template <class In, class Out>
bool Mapping<In,Out>::checkApplyJ( OutVecDeriv& out, const InVecDeriv& in, const sofa::defaulttype::BaseMatrix* J )
{
    applyJ(out, in);
    if (!J)
    {
        serr << "CheckApplyJ: getJ returned a NULL matrix" << sendl;
        return false;
    }

    behavior::BaseMechanicalState* toMechaModel = NULL;
    helper::vector<behavior::BaseMechanicalState*> toMechaModelVec = this->getMechTo();
    if(toMechaModelVec.size() < 1)
        return false;
    else toMechaModel = toMechaModelVec[0];

    if (toMechaModel->forceMask.isInUse())
    {
        serr << "Mask in use in mapped model. Disabled because of checkApplyJ." << sendl;
        toMechaModel->forceMask.setInUse(false);
    }

    OutVecDeriv out2;
    out2.resize(out.size());

    matrixApplyJ(out2, in, J);

    // compare out and out2
    const int NOut = sofa::defaulttype::DataTypeInfo<typename Out::Deriv>::Size;
    double diff_mean = 0, diff_max = 0, val1_mean = 0, val2_mean = 0;
    for (unsigned int i=0; i<out.size(); ++i)
        for (int j=0; j<NOut; ++j)
        {
            double v1 = out[i][j];
            double v2 = out2[i][j];
            double diff = v1-v2;
            if (diff < 0) diff = -diff;
            if (diff > diff_max) diff_max = diff;
            diff_mean += diff;
            if (v1 < 0) v1=-v1;
            val1_mean += v1;
            if (v2 < 0) v2=-v2;
            val2_mean += v2;
        }
    diff_mean /= out.size() * NOut;
    val1_mean /= out.size() * NOut;
    val2_mean /= out.size() * NOut;
    sout << "Comparison of applyJ() and matrix from getJ(): ";
    sout << "Max Error = " << diff_max;
    sout << "\t Mean Error = " << diff_mean;
    sout << "\t Mean Abs Value from applyJ = " << val1_mean;
    sout << "\t Mean Abs Value from matrix = " << val2_mean;
    sout << sendl;
    if (this->f_printLog.getValue() || diff_max > 0.1*(val1_mean+val2_mean)/2)
    {
        sout << "Input vector       : " << in << sendl;
        sout << "Result from applyJ : " << out << sendl;
        sout << "Result from matrix : " << out2 << sendl;
    }
    return true;
}

template <class In, class Out>
void Mapping<In,Out>::matrixApplyJ( OutVecDeriv& out, const InVecDeriv& in, const sofa::defaulttype::BaseMatrix* J )
{
    typedef typename Out::Real OutReal;
    typedef typename In::Real InReal;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    if (!J) return;
    if (J->rowSize() == 0) return;
    const int NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size;
    const int NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size;
    out.resize(J->rowSize() / NOut);
    OutReal* in_alloc = NULL;
    OutReal* out_alloc = NULL;
    const OutReal* in_buffer = NULL;
    OutReal* out_buffer = NULL;
    if (sizeof(InReal) == sizeof(OutReal) && sofa::defaulttype::DataTypeInfo<InDeriv>::SimpleLayout)
    {
        // we can use the data directly
        in_buffer = (const OutReal*)&in[0];
    }
    else
    {
        // we must copy the values
        in_alloc = new OutReal[in.size()*NIn];
        for (unsigned int i=0; i<in.size(); ++i)
            for (int j=0; j<NIn; ++j)
                in_alloc[i*NIn+j] = (OutReal)in[i][j];
        in_buffer = in_alloc;
    }
    if (sofa::defaulttype::DataTypeInfo<OutDeriv>::SimpleLayout)
    {
        // we can use the data directly
        out_buffer = (OutReal*)&out[0];
    }
    else
    {
        // we must copy the values
        out_alloc = new OutReal[out.size()*NOut];
        for (unsigned int i=0; i<out.size(); ++i)
            for (int j=0; j<NOut; ++j)
                out_alloc[i*NOut+j] = (OutReal)0; //out[i][j];
        out_buffer = out_alloc;
    }
    // Do the matrix multiplication
    J->opMulV(out_buffer, in_buffer);
    if (in_alloc)
    {
        delete[] in_alloc;
    }
    if (out_alloc)
    {
        for (unsigned int i=0; i<out.size(); ++i)
            for (int j=0; j<NOut; ++j)
                out[i][j] = out_alloc[i*NOut+j];
        delete[] out_alloc;
    }
}

template <class In, class Out>
bool Mapping<In,Out>::checkApplyJT( InVecDeriv& out, const OutVecDeriv& in, const sofa::defaulttype::BaseMatrix* J )
{
    if (!J)
    {
        serr << "CheckApplyJT: getJ returned a NULL matrix" << sendl;
        applyJT(out, in);
        return false;
    }

    behavior::BaseMechanicalState* toMechaModel = NULL;
    helper::vector<behavior::BaseMechanicalState*> toMechaModelVec = this->getMechTo();
    if(toMechaModelVec.size() < 1)
        return false;
    else toMechaModel = toMechaModelVec[0];

    if (toMechaModel->forceMask.isInUse())
    {
        serr << "Mask in use in mapped model. Disabled because of checkApplyJT." << sendl;
        toMechaModel->forceMask.setInUse(false);
    }

    InVecDeriv tmp;
    tmp.resize(out.size());
    applyJT(tmp, in);
    if (tmp.size() > out.size())
        out.resize(tmp.size());
    for (unsigned int i=0; i<tmp.size(); ++i)
        out[i] += tmp[i];

    InVecDeriv tmp2;
    tmp2.resize(out.size());

    matrixApplyJT(tmp2, in, J);

    // compare tmp and tmp2
    const int NOut = sofa::defaulttype::DataTypeInfo<typename In::Deriv>::Size;
    double diff_mean = 0, diff_max = 0, val1_mean = 0, val2_mean = 0;
    for (unsigned int i=0; i<tmp.size(); ++i)
        for (int j=0; j<NOut; ++j)
        {
            double v1 = tmp[i][j];
            double v2 = tmp2[i][j];
            double diff = v1-v2;
            if (diff < 0) diff = -diff;
            if (diff > diff_max) diff_max = diff;
            diff_mean += diff;
            if (v1 < 0) v1=-v1;
            val1_mean += v1;
            if (v2 < 0) v2=-v2;
            val2_mean += v2;
        }
    diff_mean /= tmp.size() * NOut;
    val1_mean /= tmp.size() * NOut;
    val2_mean /= tmp.size() * NOut;
    sout << "Comparison of applyJT() and matrix^T from getJ(): ";
    sout << "Max Error = " << diff_max;
    sout << "\t Mean Error = " << diff_mean;
    sout << "\t Mean Abs Value from applyJT = " << val1_mean;
    sout << "\t Mean Abs Value from matrixT = " << val2_mean;
    sout << sendl;
    if (this->f_printLog.getValue() || diff_max > 0.1*(val1_mean+val2_mean)/2)
    {
        sout << "Input vector        : " << in << sendl;
        sout << "Result from applyJT : " << tmp << sendl;
        sout << "Result from matrixT : " << tmp2 << sendl;
    }
    return true;
}

template <class In, class Out>
void Mapping<In,Out>::matrixApplyJT( InVecDeriv& out, const OutVecDeriv& in, const sofa::defaulttype::BaseMatrix* J )
{
    typedef typename Out::Real OutReal;
    typedef typename In::Real InReal;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::Deriv InDeriv;
    if (!J) return;
    if (J->rowSize() == 0) return;
    const int NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size;
    const int NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size;
    out.resize(J->colSize() / NOut);
    InReal* in_alloc = NULL;
    InReal* out_alloc = NULL;
    const InReal* in_buffer = NULL;
    InReal* out_buffer = NULL;
    if (sofa::defaulttype::DataTypeInfo<OutDeriv>::SimpleLayout)
    {
        // we can use the data directly
        in_buffer = (const InReal*)&in[0];
    }
    else
    {
        // we must copy the values
        in_alloc = new InReal[in.size()*NOut];
        for (unsigned int i=0; i<in.size(); ++i)
            for (int j=0; j<NOut; ++j)
                in_alloc[i*NOut+j] = (InReal)in[i][j];
        in_buffer = in_alloc;
    }
    if (sizeof(InReal) == sizeof(OutReal) && sofa::defaulttype::DataTypeInfo<InDeriv>::SimpleLayout)
    {
        // we can use the data directly
        out_buffer = (InReal*)&out[0];
    }
    else
    {
        // we must copy the values
        out_alloc = new InReal[out.size()*NIn];
        for (unsigned int i=0; i<out.size(); ++i)
            for (int j=0; j<NIn; ++j)
                out_alloc[i*NIn+j] = (InReal)0; //out[i][j];
        out_buffer = out_alloc;
    }
    // Do the transposed matrix multiplication
    J->opPMulTV(out_buffer, in_buffer);
    if (in_alloc)
    {
        delete[] in_alloc;
    }
    if (out_alloc)
    {
        for (unsigned int i=0; i<out.size(); ++i)
            for (int j=0; j<NIn; ++j)
                out[i][j] += out_alloc[i*NIn+j];
        delete[] out_alloc;
    }
}

template <class In, class Out>
bool Mapping<In,Out>::checkApplyJT( InMatrixDeriv& out, const OutMatrixDeriv& in, const sofa::defaulttype::BaseMatrix* J )
{
    applyJT(out, in);
    if (!J)
    {
        serr << "CheckApplyJT: getJ returned a NULL matrix" << sendl;
        return false;
    }

    return true;
}

template <class In, class Out>
void Mapping<In,Out>::matrixApplyJT( InMatrixDeriv& /*out*/, const OutMatrixDeriv& /*in*/, const sofa::defaulttype::BaseMatrix* /*J*/ )
{
    serr << "matrixApplyJT for MatrixDeriv NOT IMPLEMENTED" << sendl;
}

} // namespace core

} // namespace sofa

#endif
