#ifndef SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_INL

#include <sofa/component/constraint/UnilateralInteractionConstraint.h>
#include <sofa/defaulttype/Vec.h>
#include <GL/gl.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::addContact(bool friction, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree)
{



    // compute dt and delta
    Real delta = dot(P-Q, norm) - contactDistance;

    // Real dt = delta / dot(P-Q, norm) - dot(Pfree-Qfree, norm);

    Real deltaFree = dot(Pfree-Qfree, norm) - contactDistance;
    Real dt;
    int i = contacts.size();
    contacts.resize(i+1);
    Contact& c = contacts[i];
// for visu
    c.P = P;
    c.Q = Q;
    c.Pfree = Pfree;
    c.Qfree = Qfree;
//
    c.m1 = m1;
    c.m2 = m2;
    c.norm = norm;
    c.delta = delta;
    c.t = Deriv(norm.z(), norm.x(), norm.y());
    c.s = cross(norm,c.t);
    c.s = c.s / c.s.norm();
    c.t = cross((-norm), c.s);
    c.friction = friction;

    if (rabs(delta - deltaFree) > 0.001 * delta)
    {
        dt = delta / (delta - deltaFree);
        if (dt > 0.0 && dt < 1.0  )
        {
            sofa::defaulttype::Vector3 Qt, Pt;
            Qt = Q*(1-dt) + Qfree*dt;
            Pt = P*(1-dt) + Pfree*dt;
            c.dfree = dot(Pfree-Pt, c.norm) - dot(Qfree-Qt, c.norm);
            c.dfree_t = dot(Pfree-Pt, c.t) - dot(Qfree-Qt, c.t);
            c.dfree_s = dot(Pfree-Pt, c.s) - dot(Qfree-Qt, c.s);
            //printf("\n ! dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
        }
        else
        {

            if (deltaFree < 0.0)
            {
                dt=0.0;
                c.dfree = dot(Pfree-P, c.norm) - dot(Qfree-Q, c.norm);
                //printf("\n dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
                c.dfree_t = dot(Pfree-P, c.t) - dot(Qfree-Q, c.t);
                c.dfree_s = dot(Pfree-P, c.s) - dot(Qfree-Q, c.s);
            }
            else
            {
                dt=1.0;
                c.dfree = deltaFree;
                c.dfree_t = 0;
                c.dfree_s = 0;
            }





        }
    }
    else
    {
        dt = 0;
        c.dfree = deltaFree;
        c.dfree_t = 0;
        c.dfree_s = 0;
        //printf("\n dt = %f, c.dfree = %f, deltaFree=%f, delta = %f", dt, c.dfree, deltaFree, delta);
    }

}


template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::applyConstraint(unsigned int &contactId)
{
    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        c.id = contactId++;

        this->object1->setConstraintId(c.id);
        this->object2->setConstraintId(c.id);

        SparseVecDeriv svd1;
        SparseVecDeriv svd2;

        svd1.push_back(SparseDeriv(c.m1, -c.norm));
        svd2.push_back(SparseDeriv(c.m2, c.norm));

        c1.push_back(svd1);
        c2.push_back(svd2);

        /*
        if (c.friction)
        {
        	// Set the constraints vector for the first mechanical model using the tangente
        	svd.push_back(SparseDeriv(c.m1, -c.t));

        	// Set the constraints vector for the second mechanical model using the tangente
        	svd[0] = SparseDeriv(c.m2, c.t);

        	// Set the constraints vector for the first mechanical model using the secant
        	svd.push_back(SparseDeriv(c.m1, -c.s));

        	// Set the constraints vector for the second mechanical model the secant
        	svd[0] = SparseDeriv(c.m2, c.s);
        }
        */
    }
}

template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::getConstraintValue(double* v /*, unsigned int &numContacts*/)
{
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i]; // get each contact detected
        v[c.id] = c.dfree; // if there is not friction, dfree is the only constraint value added to v

        //	if (c.friction)
        //	{
        //		v[j+(*offset)] = c.dfree_t; // if there is friction, dfree_t & dfree_s are added to v too
        //		v[j+(*offset)] = c.dfree_s;
        //		j += 2;
        //	}
    }
}


template<class DataTypes>
void UnilateralInteractionConstraint<DataTypes>::draw()
{
    if (!getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor4f(1,0,0,1);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        glLineWidth(1);
        glColor4f(1,0,0,1);
        const Contact& c = contacts[i];
        helper::gl::glVertexT(c.P);
        helper::gl::glVertexT(c.Q);

        if (c.dfree < 0)
        {
            glLineWidth(5);
            glColor4f(0,1,0,1);
            helper::gl::glVertexT(c.Pfree);
            helper::gl::glVertexT(c.Qfree);
        }


    }
    glEnd();
    /*
    	glLineWidth(5);
    	//if (getContext()->getShowNormals())
    	{
    		glColor4f(1,1,0,1);
    		glBegin(GL_LINES);
    		for (unsigned int i=0; i<contacts.size(); i++)
    		{
    			const Contact& c = contacts[i];
    			//if (c.pen > 0) continue;
    			//std::cout << " lambda["<<i<<"]="<<lambda[i]<<std::endl;
    			Coord p = p1[c.m1] - c.norm * lambda[i];
    			helper::gl::glVertexT(p1[c.m1]);
    			helper::gl::glVertexT(p);
    			p = p2[c.m2] + c.norm * lambda[i];
    			helper::gl::glVertexT(p2[c.m2]);
    			helper::gl::glVertexT(p);
    		}
    		glEnd();
    	}
    */
    glLineWidth(1);
}



} // namespace constraint

} // namespace component

} // namespace sofa

#endif
