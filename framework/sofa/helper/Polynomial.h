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
#ifndef SOFA_HELPER_POLYNOMIAL_H
#define SOFA_HELPER_POLYNOMIAL_H

#include <sofa/helper/helper.h>

//#include <sofa/component/component.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <stdarg.h>
#include <iostream>
#include <string>
//#include <vector>
#include <list>
#include <cmath>

namespace sofa
{
namespace helper
{


using namespace sofa::defaulttype;
using namespace std;
/**
 * \file modules/sofa/component/femToolsForFEM.h
 * \namespace sofa::component::fem
 * \brief Tools used in FEM computing
 *
 */
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

/**
 * \class template<typename Real, unsigned int N>Monomial
 *
 * A generic monomial with <N> variable of type <Real>
 * A monomial is one term in the polynomial
 *
 * Todo this first version of polynomial is a linear presentation.
 * This presentation is efficient for arithmetic operator like +,-,\*,/ and derivative.
 * But is not optimized for the evaluating (comparing to the recurent presentation).
 */
template<typename Real, unsigned int N>
class Monomial
{

public :

    typedef sofa::defaulttype::Vec<N,Real> RNpoint;
    ///to define the derivative operator m_mono.d(x), m_mono.d(y)...
    enum {x,y,z,xy,yz,zx};

    Real coef;
    sofa::defaulttype::Vec<N,int>  powers;

    Monomial();
    Monomial(const Monomial<Real,N> & a);
    Monomial(const Real &,...);
    Monomial<Real,N>& operator=(const Monomial<Real,N> & b);

    ///Setting of Monomial
    void Set(const Real &,...);
    void SetCoef (const Real & m_coef) {coef=m_coef;}
    void SetPower(sofa::helper::vector<int> & m_powers)   {for(unsigned int i=0; i<N; i++) powers[i]=m_powers[i];}
    void SetPower(const int & numbervar,const int &powervalue) {powers[numbervar]=powervalue;}

    ///Return the total degree of monomial
    int degree();

    ///Logical operators
    bool operator ==(const Monomial<Real,N> & b) const ;
    bool operator !=(const Monomial<Real,N> & b) const {return !(*this == b);}
    bool isSamePowers(const Monomial<Real,N> & b) const;
    bool isNULL() const {return (coef == (Real) 0.);}

    ///Mathematical operators
    Monomial<Real,N> & operator*=(const Real & alpha) {this->coef*=alpha; return *this;}
    Monomial<Real,N> & operator/=(const Real & alpha) {this->coef/=alpha; return *this;}
    Monomial<Real,N> & operator+=(const Monomial<Real,N> & b);
    Monomial<Real,N> & operator-=(const Monomial<Real,N> & b);
    Monomial<Real,N> & operator*=(const Monomial<Real,N> & b);
    Monomial<Real,N> operator+ () const {Monomial<Real,N> r(*this); return r;}
    Monomial<Real,N> operator- () const {Monomial<Real,N> r(*this); r*=(Real) -1.; return r;}

    Monomial<Real,N> operator*(const Real & alpha) {Monomial<Real,N> r(*this); r*=alpha; return r;}

    Monomial<Real,N> operator/(const Real & alpha) {Monomial<Real,N> r(*this); r/=alpha; return r;}
    Monomial<Real,N> operator+(const Monomial<Real,N> & a) {Monomial<Real,N> r(*this); r+=a; return r;}
    Monomial<Real,N> operator-(const Monomial<Real,N> & a) {Monomial<Real,N> r(*this); r-=a; return r;}
    Monomial<Real,N> operator*(const Monomial<Real,N> & a) {Monomial<Real,N> r(*this); r*=a; return r;}

    ///Evaluating value
    Real operator()(const sofa::helper::vector<Real> & x) const;
    Real operator()(const RNpoint & x) const;
    ///Evaluating derivative value
    Real operator()(const sofa::helper::vector<Real> & x,unsigned int ideriv) const;
    Real operator()(const RNpoint & x,unsigned int ideriv) const;

    ///Derivative operator alowing to write p1=p2.d(x);
    Monomial<Real,N> d(const unsigned int & ideriv) const;

    void writeToStream(std::ostream & ff) const;
    void readFromStream(std::istream & ff);

    template<typename FReal, unsigned int FN>
    inline friend std::ostream & operator <<(std::ostream & out,const Monomial<FReal,FN> & m_monomial)
    {m_monomial.writeToStream(out); return out;}

    template<typename FReal, unsigned int FN>
    inline friend std::istream & operator <<(std::istream & in, Monomial<FReal,FN> & m_monomial)
    {m_monomial.readFromStream(in); return in;}

    template<typename FReal, unsigned int FN> //For comutativity of operator *: Monomial*Real || Real*Monomial.
    friend Monomial<FReal,FN> & operator*(const FReal & alpha,Monomial<FReal,FN> & r);

protected :
    sofa::defaulttype::Vec<N,string> variables;
};



/**
 * \class template<typename Real, unsigned int N>Polynomial
 *
 * A generic polynomial with <N> variable of type <Real>
 * A polynomial is a list composed several term of monomial
 *
 */

template<typename Real, unsigned int N>
class SOFA_HELPER_API Polynomial
{

public :

    typedef list< Monomial<Real,N> > MonomialsList;
    typedef typename MonomialsList::const_iterator MonomialConstIterator;
    typedef typename MonomialsList::iterator MonomialIterator;
    typedef sofa::defaulttype::Vec<N,Real> RNpoint;

    sofa::helper::vector< Monomial<Real,N> > listofTerms;

    ///Default constructor
    Polynomial();

    ///Copy constructor
    Polynomial(const Polynomial<Real,N> & a);
    Polynomial(const Monomial<Real,N> & a);

    ///constructor
    /// \exemple Polynomial<Real,2> p(5,Real,int, int,Real,int, int,Real,int, int,Real,int, int,Real,int, int)
    Polynomial(const unsigned int & nbofTerm,...);

    ///Assign operator
    Polynomial<Real,N> & operator=(const Polynomial<Real,N> & b) {listofTerms=b.listofTerms; return *this;}

    int degree();

    ///Return true if a and b has the same powers
    bool operator ==(const Polynomial<Real,N> & b) const ;
    bool operator !=(const Polynomial<Real,N> & b) const {return !((*this) == b);}

    Polynomial<Real,N>  & operator*=(const Real & alpha) ;
    Polynomial<Real,N>  & operator/=(const Real & alpha) ;
    Polynomial<Real,N>  & operator+=(const Monomial<Real,N> & b) ;
    Polynomial<Real,N>  & operator+=(const Polynomial<Real,N> & b) ;
    Polynomial<Real,N>  & operator-=(const Polynomial<Real,N> & b) ;
    Polynomial<Real,N>   operator+ () const {Polynomial<Real,N> r(*this); return r;}
    Polynomial<Real,N>   operator- () const;

    Polynomial<Real,N>  & operator*=(const Polynomial<Real,N> & b);

    Polynomial<Real,N>  operator*(          const Real & alpha) {Polynomial<Real,N> r(*this); r*=alpha; return r;}
    Polynomial<Real,N>  operator/(          const Real & alpha) {Polynomial<Real,N> r(*this); r/=alpha; return r;}
    Polynomial<Real,N>  operator+(const Monomial<Real,N>   & a) {Polynomial<Real,N> r(*this); r+=a; return r;}
    Polynomial<Real,N>  operator+(const Polynomial<Real,N> & a) {Polynomial<Real,N> r(*this); r+=a; return r;}
    Polynomial<Real,N>  operator-(const Polynomial<Real,N> & a) {Polynomial<Real,N> r(*this); r-=a; return r;}
    Polynomial<Real,N>  operator*(const Polynomial<Real,N> & a) {Polynomial<Real,N> r(*this); r*=a; return r;}

    ///Evaluating
    //Real operator()(const sofa::defaulttype::Vec<N,Real> & x) const;
    Real operator()(const RNpoint & x) const;
    ///Evaluating derivative
    //Real operator()(const sofa::defaulttype::Vec<N,Real> & x,unsigned int iderive) const;
    Real operator()(const RNpoint & x,unsigned int iderive) const;

    ///Derivative operator alowing to write p1=p2.d(x);
    Polynomial<Real,N>  d(const unsigned int & ideriv) const;

    void writeToStream(ostream & stream) const;

    ///Comutativity of operator*(Real):
    ///Allowing to write p1=r*p2;   or   p1=p2*r;
    ///Polynomial =  Polynomial*Real || Real*Polynomial.
    template<typename FReal, unsigned int FN>
    friend Polynomial<FReal,FN> & operator*(const FReal & alpha,Polynomial<FReal,FN> & r);

    /// poly=poly+mono || mono + poly
    template<typename FReal, unsigned int FN>
    friend Polynomial<FReal,FN> & operator*(const Monomial<FReal,FN>   & a, Polynomial<FReal,FN> & r);

protected :

    ///The two sort will help to transform Linear presentation to the recurent one and reciprocal.
    void LinearSort();  // don't forget to erase all null coef term
    void RecurentSort();// don't forget to erase all null coef term
};
/*

////////////////////////////////
template<typename FReal, unsigned int FN>
inline ostream & operator <<(ostream & f, const Monomial<FReal,FN> & m_monomial )
{
	m_monomial.printToStream(f);
	return f;
}

template<typename FReal, unsigned int FN>
inline ostream & operator<<(ostream & f, const Polynomial<FReal,FN> & m_polynomial )
{
	m_polynomial.printToStream(f);
	return f;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
template<class TDataTypes>
void TestPolynomial()
{
	typedef TDataTypes DataTypes;
	typedef typename DataTypes::Coord Coord;
	typedef typename Coord::value_type Real;


	sofa::component::fem::Monomial<Real,3> m;
	sofa::component::fem::Monomial<Real,3> m1(1.,1,2,3);
	sofa::component::fem::Monomial<Real,3> m2(2.,4,5,6);


	m.Set(5.,8,6,9);m.SetCoef(5.);    m.SetPower(2,4);
	if (m.isNULL()) cout<<"TRUE"<<endl; else cout<<"FALSE"<<endl;
	m=m1;m.SetCoef(100.);
	if (m.isNULL()) cout<<"TRUE"<<endl; else cout<<"FALSE"<<endl;



	if (p1==p2) cout<<"(p1==p2) : TRUE"<<endl; else cout<<"(p1==p2) : FALSE"<<endl;
	p1=p2;
	if (p1==p2) cout<<"(p1==p2) : TRUE"<<endl; else cout<<"(p1==p2) : FALSE"<<endl;



	//sofa::component::fem::Polynomial<Real,3> p2=p.d(1);
	//int * power = {1,4,5};
	//vector<int> powers;powers.push_back(5);powers.push_back(10);powers.push_back(15);
	vector<Real> x;x.push_back(2.);x.push_back(2.);x.push_back(5.);
	//power[0]=1;power[1]=4;power[0]=5;////////////////////////////////////////
	//p.SetCoef(100.);
	//p.SetPower(powers);
	//cout<<m<<"  "<<m.degree()<<endl;cout<<m1<<"  "<<m1.degree()<<endl;cout<<m2<<"  "<<m2.degree()<<endl;
	//cout<<"===================================================="<<endl<<endl;
	//m2=m1;
	//m2.SetPower(2,3);
	//m1=m2.d(2);
	//cout<<"m2 :"<<m2<<endl;
	//cout<<"m1 :"<<m1<<"  "<<m1.degree()<<"     "<<m1(x)<<"        "<<m1(2,x)<<endl;


	sofa::component::fem::Monomial<Real,3> mono(1.,2,4,8);
	sofa::component::fem::Polynomial<Real,3> p(mono);
	sofa::component::fem::Polynomial<Real,3> p1(3,-1.,1,1,1,2.,2,2,2,-3.,3,3,3);
	//sofa::component::fem::Polynomial<Real,3> p1(2,-1.,1,1,1,2.,2,2,2);
	sofa::component::fem::Polynomial<Real,3> p2(3,1.,1,2,3,4.,4,5,6,10.,10,20,30);

	vector<Real> x;x.push_back(10.);x.push_back(1.);x.push_back(2.);

	cout<<"p   :"<<p <<"   p.degree()   "<< p.degree()<<endl;
	cout<<"p1  :"<<p1<<"   p1.degree()  "<< p1.degree()<<endl;
	cout<<"p2  :"<<p2<<"   p2.degree()  "<< p2.degree()<<endl;
	cout<<"================================================================="<<endl<<endl;

	int iderive=2;
	//p1 = p2.d(iderive);
	cout<<"p1  :"<<p1<<"   p1.degree()  "<< p1.degree()<<endl
			 <<"   evaluating  "<<p1(x)<<"  evaluating derivative     "<<p1(iderive,x)<<endl;


}
*/
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////




} // namespace helper

} // namespace sofa

#endif

