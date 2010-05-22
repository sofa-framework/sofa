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
#ifndef SOFA_CORE_MULTI2MAPPING_H
#define SOFA_CORE_MULTI2MAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/core.h>
#include <sofa/core/VecId.h>


namespace sofa
{

namespace core
{

/**
 *  \brief Specialized interface to describe many to many mapping.
 *   Currently supports only one type for inputs and one type for outputs.
 */

template <class TIn1, class TIn2, class TOut>
class Multi2Mapping : public BaseMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(Multi2Mapping,TIn1, TIn2,TOut), BaseMapping);

    /// Input Model Type
    typedef TIn1 In1;
    typedef TIn2 In2;
    /// Output Model Type
    typedef TOut Out;

    typedef typename In1::VecCoord In1VecCoord;
    typedef typename In1::VecDeriv In1VecDeriv;
    typedef typename In2::VecCoord In2VecCoord;
    typedef typename In2::VecDeriv In2VecDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;

protected:
    /// Input Models container. New inputs are added through addInputModel(In* ).
    helper::vector<In1*> fromModels1;
    helper::vector<In2*> fromModels2;
    /// Output Model container. New outputs are added through addOutputModel( Ou* )
    helper::vector<Out*> toModels;
public:
    /// Constructor
    Multi2Mapping() {} ;
    /// Destructor
    virtual ~Multi2Mapping() {};

    virtual void addInputModel(In1* );
    virtual void addInputModel(In2* );
    virtual void addOutputModel( Out* );

    /// Return the reference to fromModels.
    template <class T>
    helper::vector<T*>& getFromModels();
    /// Return reference to toModels.
    helper::vector<Out*>& getToModels();

    /// Return a container of input models statically casted as BaseObject*
    helper::vector<objectmodel::BaseObject* > getFrom();
    /// Return container of output model statically casted as BaseObject*.
    helper::vector<objectmodel::BaseObject* > getTo();

    /// Apply the mapping on position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    ///
    /// This method must be reimplemented by all mappings.
    /// InPos and OutPos by default contains VecIds of type V_COORD.
    /// The size of InPos vector is the same as the number of fromModels.
    /// The size of OutPos vector is the same as the number of OutModels.
    virtual void apply(const helper::vector<OutVecCoord*>& outPos, const helper::vector<const In1VecCoord*>& inPos1 , const helper::vector<const In2VecCoord*>& inPos2 ) = 0;

    /// Apply the mapping on derived (velocity, displacement) vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    ///
    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const In1VecDeriv*>& inDeriv1, const helper::vector<const In2VecDeriv*>& inDeriv2) = 0;

    virtual void init();

    /// Apply the mapping to position and velocity vectors.
    ///
    /// This method call the internal apply(helper::vector<VecId>& inPos, helper::vector<VecId>& outPos)
    /// and applyJ(helper::vector<VecId>& inDeriv, helper::vector<VecId>& outDeriv) methods.
    virtual void updateMapping();

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    virtual void disable();



    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


    static std::string templateName(const Multi2Mapping<TIn1,TIn2, TOut>* = NULL);

protected:
    void getVecIn1Coord     (const VecId &id, helper::vector<      In1VecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(fromModels1[i]->getVecCoord(id.index));    }
    void getConstVecIn1Coord(const VecId &id, helper::vector<const In1VecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(fromModels1[i]->getVecCoord(id.index));    }

    void getVecIn1Deriv     (const VecId &id, helper::vector<      In1VecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(fromModels1[i]->getVecDeriv(id.index));    }
    void getConstVecIn1Deriv(const VecId &id, helper::vector<const In1VecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(fromModels1[i]->getVecDeriv(id.index));    }

    void getVecIn2Coord     (const VecId &id, helper::vector<      In2VecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecCoord(id.index));    }
    void getConstVecIn2Coord(const VecId &id, helper::vector<const In2VecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecCoord(id.index));    }

    void getVecIn2Deriv     (const VecId &id, helper::vector<      In2VecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecDeriv(id.index));    }
    void getConstVecIn2Deriv(const VecId &id, helper::vector<const In2VecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecDeriv(id.index));    }

    void getVecOutCoord     (const VecId &id, helper::vector<      OutVecCoord*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecCoord(id.index));    }
    void getConstVecOutCoord(const VecId &id, helper::vector<const OutVecCoord*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecCoord(id.index));    }

    void getVecOutDeriv     (const VecId &id, helper::vector<      OutVecDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecDeriv(id.index));    }
    void getConstVecOutDeriv(const VecId &id, helper::vector<const OutVecDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecDeriv(id.index));    }


    /// If true, display the mapping
    bool getShow() const { return this->getContext()->getShowMappings(); }
};

} // namespace core

} // namespace sofa

#endif

