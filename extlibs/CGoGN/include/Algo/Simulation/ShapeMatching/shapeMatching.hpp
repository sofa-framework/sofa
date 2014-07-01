/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Simulation
{

namespace ShapeMatching
{


template <typename PFP>
ShapeMatching<PFP>::ShapeMatching(MAP& map, VertexAttribute<VEC3>& position, VertexAttribute<REAL>& mass):
    m_map(map),
    m_position(position),
    m_mass(mass)
{
    unsigned int nbE = m_position.nbElements();

    m_q.reserve(nbE);

    m_goal = this->m_map.template getAttribute<VEC3, VERTEX>("goal");

    if(!m_goal.isValid())
        m_goal = this->m_map.template addAttribute<VEC3, VERTEX>("goal");
}

template <typename PFP>
ShapeMatching<PFP>::~ShapeMatching()
{
    if(m_goal.isValid())
      m_map.template removeAttribute<VEC3, VERTEX>(m_goal);
}

template <typename PFP>
Eigen::Vector3d ShapeMatching<PFP>::massCenter() //
{
    Eigen::Vector3d xcm = Eigen::Vector3d::Zero();
    REAL m = 0.0;

    for(unsigned int i = m_position.begin() ; i < m_position.end() ; m_position.next(i))
    {
        Eigen::Vector3d tmp ;
        for (unsigned int j = 0 ; j < 3 ; ++j)
            tmp(j) = m_position[i][j] ;

        xcm += m_mass[i] * tmp;
        m += m_mass[i];
    }

    xcm /= m;

    return xcm;
}

/**
 * Initialize pre-computed ....
 * First, \f$ x^{0}_{cm} \f$
 * In a second step, \f$ q_{i} = x^{0}_{i} - x^{0}_{cm} \f$
 */
template <typename PFP>
void ShapeMatching<PFP>::initialize()
{
    Eigen::Vector3d x0cm = massCenter();

    for(unsigned int i = m_position.begin() ; i < m_position.end() ; m_position.next(i))
    {
        Eigen::Vector3d tmp ;
        for (unsigned int j = 0 ; j < 3 ; ++j)
            tmp(j) = m_position[i][j] ;

        Eigen::Vector3d qi = tmp - x0cm;

        m_q.push_back(qi); //q_{i} = x^{0}_{i} - x^{0}_{cm}
    }
}

template <typename PFP>
void ShapeMatching<PFP>::shapeMatch()
{
    // p_{i}
    std::vector<Eigen::Vector3d> m_p;

    m_p.reserve(m_position.nbElements());

    //1.
    Eigen::Vector3d xcm = massCenter();

    for(unsigned int i = m_position.begin() ; i < m_position.end() ; m_position.next(i))
    {
        Eigen::Vector3d tmp ;
        for (unsigned int j = 0 ; j < 3 ; ++j)
            tmp(j) = m_position[i][j] ;

        Eigen::Vector3d pi = tmp - xcm ;

        m_p.push_back(pi) ; //p_{i} = x_{i} - x_{cm}

    }

    //2.
    Eigen::Matrix3d apq = Eigen::Matrix3d::Zero();

    for(unsigned int i=0 ; i < m_p.size() ; ++i)
    {
        apq(0,0) += m_p[i][0] * m_q[i][0];
        apq(0,1) += m_p[i][0] * m_q[i][1];
        apq(0,2) += m_p[i][0] * m_q[i][2];

        apq(1,0) += m_p[i][1] * m_q[i][0];
        apq(1,1) += m_p[i][1] * m_q[i][1];
        apq(1,2) += m_p[i][1] * m_q[i][2];

        apq(2,0) += m_p[i][2] * m_q[i][0];
        apq(2,1) += m_p[i][2] * m_q[i][1];
        apq(2,2) += m_p[i][2] * m_q[i][2];
    }

    Eigen::Matrix3d S = apq.transpose() * apq ; //symmetric matrix

    //3. Jacobi Diagonalisation
    Eigen::EigenSolver<Eigen::Matrix3d> es(S);

    //V * D * V^(-1)
    Eigen::Matrix3d D =  es.pseudoEigenvalueMatrix();
    Eigen::Matrix3d U =  es.pseudoEigenvectors() ;

    for(int j = 0; j < 3; j++)
    {
        if(D(j,j) <= 0)
        {
            D(j,j) = 0.05f;
        }
        D(j,j) = 1.0f/sqrt(D(j,j));
    }

    S = U * D * U.transpose();

    // Now we can get the rotation part
    Eigen::Matrix3d R = apq * S; //S^{-1}

    //4.
    for(unsigned int i = m_goal.begin() ; i < m_goal.end() ; m_goal.next(i))
    {
       Eigen::Vector3d tmp = R * m_q[i] + xcm; // g_{i} = R * q_i + x_{cm}

       VEC3 g;
       for (unsigned int j = 0 ; j < 3 ; ++j)
            g[j] = tmp(j);

       m_goal[i] = g;
    }
}


// \alpha : stiffness | v_i : velocity | f_ext : force exterieure
template <typename PFP>
void ShapeMatching<PFP>::computeVelocities(VertexAttribute<VEC3>& velocity, VertexAttribute<VEC3>& fext, REAL h, REAL alpha)
{
    for(unsigned int i = velocity.begin() ; i < velocity.end() ; velocity.next(i))
    {
        velocity[i] = velocity[i] + alpha * ((m_goal[i] - m_position[i]) / h ) + (h * fext[i]) / m_mass[i];
    }
}

template <typename PFP>
void ShapeMatching<PFP>::applyVelocities(VertexAttribute<VEC3>& velocity, REAL h)
{
    for(unsigned int i = m_position.begin() ; i < m_position.end() ; m_position.next(i))
    {
        m_position[i] = m_position[i] + h * velocity[i];
    }
}


} // namespace ShapeMatching

} // namespace Simulation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
