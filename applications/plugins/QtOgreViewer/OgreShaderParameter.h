/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef OGRESHADERPARAMETER_H
#define OGRESHADERPARAMETER_H

#include <sofa/helper/fixed_array.h>
#include <sofa/core/visual/VisualModel.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{


class BaseOgreShaderParameter :public core::visual::VisualModel
{
public:
    SOFA_CLASS(BaseOgreShaderParameter, core::visual::VisualModel);

    BaseOgreShaderParameter():
        entryPoint(initData(&entryPoint,5,"entryPoint", "Entry Point for the parameter:\nthe first 4 entries are automatically binded with ambient, diffuse, specular and shininess"))
        , dirty(true)
    {
    };
    virtual ~BaseOgreShaderParameter() {};

    int getEntryPoint() const {return entryPoint.getValue();};
    void setEntryPoint(int e) {entryPoint.setValue(e);};

    virtual unsigned int getValueSize() const=0;
    virtual float getValue(unsigned int index) const =0;
    virtual bool isDirty() const {return dirty;}
    virtual void reinit() {dirty=true;};



    template <class T> void setValue(T& value)
    {
        //assert(value.size() >=4);

        unsigned int i=0;
        for (i=0; i<getValueSize(); ++i) setValue(value[i],i);

        dirty = true;
    }
    template <class T> void getValue(T& value) const
    {
        //assert(value.size() >=4);

        unsigned int i=0;
        for (i=0; i<getValueSize(); ++i) value[i]=getValue(i);
        for (; i<4; ++i)                 value[i]=0.0f;

        dirty = false;
    }


protected:
    Data<int> entryPoint;
    mutable bool dirty;
};

template <unsigned int N>
class OgreShaderParameter : public BaseOgreShaderParameter
{

public:
    SOFA_CLASS(OgreShaderParameter, BaseOgreShaderParameter);

    OgreShaderParameter():value(initData(&value,"value", "Value for the Shader Parameter"))
    {
    };

    unsigned int getValueSize() const {return N;};
    static unsigned int size() {return N;}
    float getValue(unsigned int index) const
    {
        if (index >= N) return 0.0;
        else            return value.getValue()[index];
    }

    void setValue(float v, unsigned int index)
    {
        if (index >= N) return;
        else
        {
            (*value.beginEdit())[index] = v;
            value.endEdit();
        }
    }


    /// Get the template type names (if any) used to instantiate this object
    virtual std::string getTemplateName() const
    {
        std::ostringstream o;
        o << N;
        return o.str();
    }
    static std::string templateName(const OgreShaderParameter<N>* = NULL)
    {
        std::ostringstream o;
        o << N;
        return o.str();
    }
protected:
    Data<helper::fixed_array<float, N> > value;
};

}
}
}

#endif

