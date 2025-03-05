/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/core/objectmodel/Base.h>
#include <sofa/component/engine/transform/TransformEngine.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/rmath.h> //R_PI

#include <cassert>

namespace sofa::component::engine::transform
{

template <class DataTypes>
TransformEngine<DataTypes>::TransformEngine()
    : f_inputX ( initData (&f_inputX, "input_position", "input array of 3d points") )
    , f_outputX( initData (&f_outputX, "output_position", "output array of 3d points") )
    , translation(initData(&translation, type::Vec3(0_sreal,0_sreal,0_sreal),"translation", "translation vector (x,y,z)") )
    , rotation(initData(&rotation, type::Vec3(0_sreal,0_sreal,0_sreal), "rotation", "rotation vector (x,y,z)") )
    , quaternion(initData(&quaternion, type::Quat<SReal>(0_sreal,0_sreal,0_sreal,1_sreal), "quaternion", "rotation quaternion (qx,qy,qz,qw)") )
    , scale(initData(&scale, type::Vec3(1_sreal,1_sreal,1_sreal),"scale", "scale factor") )
    , inverse(initData(&inverse, false, "inverse", "true to apply inverse transformation"))
{
    addInput(&f_inputX);
    addInput(&translation);
    addInput(&rotation);
    addInput(&quaternion);
    addInput(&scale);
    addInput(&inverse);
    addOutput(&f_outputX);
    setDirtyValue();
}

template <class DataTypes>
void TransformEngine<DataTypes>::init()
{
}

template <class DataTypes>
void TransformEngine<DataTypes>::reinit()
{
    update();
}

//Declare a TransformOperation class able to do an operation on a Coord
template <class DataTypes>
struct TransformOperation
{
    virtual ~TransformOperation() {}
    virtual void execute(typename DataTypes::Coord &v) const =0;
};

//*****************************************************************
//Scale Operation
template <class DataTypes>
struct Scale : public TransformOperation<DataTypes>
{
    typedef typename DataTypes::Real Real;
    Scale():sx(0),sy(0),sz(0) {}

    void execute(typename DataTypes::Coord &p) const
    {
        Real x,y,z;
        DataTypes::get(x,y,z,p);
        DataTypes::set(p,x*sx,y*sy,z*sz);
    }

    void configure(const type::Vec3 &s, bool inverse)
    {
        if (inverse)
        {
            sx=(Real)(1.0/s[0]); sy=(Real)(1.0/s[1]); sz=(Real)(1.0/s[2]);
        }
        else
        {
            sx=(Real)s[0]; sy=(Real)s[1]; sz=(Real)s[2];
        }
    }
private:
    Real sx,sy,sz;
};


//*****************************************************************
//Rotation Operation
template <class DataTypes, int N, bool isVector>
struct RotationSpecialized : public TransformOperation<DataTypes>
{
	typedef typename DataTypes::Real Real;

    void execute(typename DataTypes::Coord &p) const
    {
        type::Vec3 pos;
        DataTypes::get(pos[0],pos[1],pos[2],p);
        pos=q.rotate(pos);
        DataTypes::set(p,pos[0],pos[1],pos[2]);
    }

    void configure(const type::Vec3 &r, bool inverse)
    {
        q=type::Quat<Real>::createQuaterFromEuler( r*(R_PI/180.0));
        if (inverse)
            q = q.inverse();
    }

    void configure(const type::Quat<SReal> &qi, bool inverse, sofa::core::objectmodel::Base*)
    {
        q=qi;
        if (inverse)
            q = q.inverse();
    }

private:
    type::Quat<SReal> q;
};

template <class DataTypes>
struct RotationSpecialized<DataTypes, 2, false> : public TransformOperation<DataTypes>
{
    typedef typename DataTypes::Real Real;

    void execute(typename DataTypes::Coord &p) const
    {
        type::Vec3 pos;
        DataTypes::get(pos[0],pos[1],pos[2],p);
        pos=q.rotate(pos);
        DataTypes::set(p,pos[0],pos[1],pos[2]);

		p.getOrientation() += rotZ;
    }

    void configure(const type::Vec3 &r, bool inverse)
    {
        q=type::Quat<Real>::createQuaterFromEuler( r*(R_PI/180.0));
		rotZ = static_cast<Real>(r.z() * (R_PI/180.0f));
        if (inverse)
            rotZ = -rotZ;
    }

    void configure(const type::Quat<SReal> &/*qi*/, bool /*inverse*/, sofa::core::objectmodel::Base* pBase)
    {
        assert(pBase);
        msg_error(pBase) << "'void RotationSpecialized::configure(const type::Quat<SReal> &qi, bool inverse)' is not implemented for two-dimensional data types";
        assert(false && "This method should not be called without been implemented");
    }

private:
    Real rotZ;
	type::Quat<SReal> q;
};

template <class DataTypes>
struct RotationSpecialized<DataTypes, 3, false> : public TransformOperation<DataTypes>
{
    typedef typename DataTypes::Real Real;

    void execute(typename DataTypes::Coord &p) const
    {
		p.getCenter() = q.rotate(p.getCenter());
		p.getOrientation() = q*p.getOrientation();
    }

    void configure(const type::Vec3 &r, bool inverse)
    {
        q=type::Quat<Real>::createQuaterFromEuler( r*(R_PI/180.0));
        if (inverse)
            q = q.inverse();
    }

    void configure(const type::Quat<SReal> &qi, bool inverse, sofa::core::objectmodel::Base*)
    {
        q=qi;
        if (inverse)
            q = q.inverse();
    }
private:
	type::Quat<SReal> q;
};

template <class DataTypes>
struct Rotation : public RotationSpecialized<DataTypes, DataTypes::spatial_dimensions, DataTypes::coord_total_size == DataTypes::spatial_dimensions>
{ };

//*****************************************************************
//Translation Operation
template <class DataTypes>
struct Translation : public TransformOperation<DataTypes>
{
    typedef typename DataTypes::Real Real;
    Translation():tx(0),ty(0),tz(0) {}
    void execute(typename DataTypes::Coord &p) const
    {
        Real x,y,z;
        DataTypes::get(x,y,z,p);
        DataTypes::set(p,x+tx,y+ty,z+tz);
    }
    void configure(const type::Vec3 &t, bool inverse)
    {
        if (inverse)
        {
            tx=(Real)-t[0]; ty=(Real)-t[1]; tz=(Real)-t[2];
        }
        else
        {
            tx=(Real)t[0]; ty=(Real)t[1]; tz=(Real)t[2];
        }
    }
private:
    Real tx,ty,tz;
};


//*****************************************************************
//Functor to apply the operations wanted
template <class DataTypes>
struct Transform
{
    typedef TransformOperation<DataTypes> Op;

    template <class  Operation>
    Operation* add(Operation *op, bool inverse)
    {
//     Operation *op=new Operation();
        if (inverse)
            list.push_front(op);
        else
            list.push_back(op);
        return op;
    }

    std::list< Op* > &getOperations() {return list;}

    void operator()(typename DataTypes::Coord &v) const
    {
        for (typename std::list< Op* >::const_iterator it=list.begin(); it != list.end() ; ++it)
        {
            (*it)->execute(v);
        }
    }
private:
    std::list< Op* > list;
};


template <class DataTypes>
void TransformEngine<DataTypes>::doUpdate()
{
    const type::Vec3 &s=scale.getValue();
    const type::Vec3 &r=rotation.getValue();
    const type::Vec3 &t=translation.getValue();
    const type::Quat<SReal> &q=quaternion.getValue();

    //Create the object responsible for the transformations
    Transform<DataTypes> transformation;
    const bool inv = inverse.getValue();
    if (s != type::Vec3(1_sreal,1_sreal,1_sreal))
        transformation.add(new Scale<DataTypes>, inv)->configure(s, inv);

    if (r != type::Vec3(0_sreal,0_sreal,0_sreal))
        transformation.add(new Rotation<DataTypes>, inv)->configure(r, inv);

    if (q != type::Quat<SReal>(0_sreal,0_sreal,0_sreal,1_sreal))
        transformation.add(new Rotation<DataTypes>, inv)->configure(q, inv, this);

    if (t != type::Vec3(0_sreal,0_sreal,0_sreal))
        transformation.add(new Translation<DataTypes>, inv)->configure(t, inv);

    //Get input
    const VecCoord& in = f_inputX.getValue();
    VecCoord& out = *(f_outputX.beginWriteOnly());

    //Set Output
    out.resize(in.size());
    //Set the output to the input
    std::copy(in.begin(),in.end(), out.begin());
    //Apply the transformation of the output
    std::for_each(out.begin(), out.end(), transformation);

    //Deleting operations
    std::list< TransformOperation<DataTypes>* > operations=transformation.getOperations();
    while (!operations.empty())
    {
        delete operations.back();
        operations.pop_back();
    }
    f_outputX.endEdit();
}



} //namespace sofa::component::engine::transform
