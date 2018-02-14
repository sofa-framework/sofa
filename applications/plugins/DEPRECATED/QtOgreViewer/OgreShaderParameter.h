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
    Data<int> entryPoint; ///< Entry Point for the parameter: the first 4 entries are automatically binded with ambient, diffuse, specular and shininess
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

