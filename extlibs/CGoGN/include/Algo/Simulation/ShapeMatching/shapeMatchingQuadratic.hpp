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
void ShapeMatchingQuadratic<PFP>::initialize()
{
    //compute q : needed to compute R

    Eigen::Vector3d x0cm = this->ShapeMatching<PFP>::massCenter();

    //compute q~
    for(unsigned int i = this->m_position.begin() ; i < this->m_position.end() ; this->m_position.next(i))
    {
        Eigen::Vector3d tmp ;
        for (unsigned int j = 0 ; j < 3 ; ++j)
            tmp(j) = this->m_position[i][j] ;

        Eigen::Vector3d qi = tmp - x0cm;

        this->m_q.push_back(qi);

        Vec9d m_qitild;

        m_qitild(0) = qi(0);
        m_qitild(1) = qi(1);
        m_qitild(2) = qi(2);

        m_qitild(3) = qi(0) * qi(0) ;
        m_qitild(4) = qi(1) * qi(1) ;
        m_qitild(5) = qi(2) * qi(2) ;

        m_qitild(6) = qi(0) * qi(1) ;
        m_qitild(7) = qi(1) * qi(2) ;
        m_qitild(8) = qi(2) * qi(0) ;

        m_qtild.push_back(m_qitild);
    }

    //compute Aqq~
    m_aqqtild = Matrix9d::Zero();

    for(unsigned int i = 0 ; i < m_qtild.size() ; ++i)
        for(int x=0;x<9;++x)
            for(int y=0;y<9;++y)
                m_aqqtild(x,y) += this->m_mass[i] * m_qtild[i][x] * m_qtild[i][y];

    Eigen::FullPivLU<Matrix9d> lu(m_aqqtild);
    m_aqqtild = lu.inverse();

//    std::cout << "m_aqqtild isInvert = " << lu.isInvertible()  << std::endl;
//    std::cout << "m_aqqtild det = " << lu.determinant() << std::endl;
//    std::cout << "apres inversion m_aqqtild = " << m_aqqtild << std::endl;
}

template <typename PFP>
void ShapeMatchingQuadratic<PFP>::shapeMatch()
{
    // p_{i}
    std::vector<Eigen::Vector3d> m_p;

    m_p.reserve(this->m_position.nbElements());

    //1.
    Matrix39d Apqtild;
    Apqtild.setZero(3,9);

    //1.bis needed to compute R
    Eigen::Vector3d xcm = this->massCenter();


    Eigen::Matrix3d apq = Eigen::Matrix3d::Zero();
    for(unsigned int i = this->m_position.begin() ; i < this->m_position.end() ; this->m_position.next(i))
    {
       //this->m_p[i] = VEC3(this->m_position[i] - xcm); //p_{i} = x_{i} - x_{cm}

        Eigen::Vector3d tmp ;
        for (unsigned int j = 0 ; j < 3 ; ++j)
            tmp(j) = this->m_position[i][j] ;

       Eigen::Vector3d pi = tmp - xcm; //p_{i} = x_{i} - x_{cm}

       m_p.push_back(pi) ;

        //2.
       for(unsigned int x=0; x<3; x++ )
           for(unsigned int y=0; y<9; y++ )
           {
               if(y<3)
                   apq(x,y) += m_p[i][x] * this->m_q[i][y];

               Apqtild(x,y) += m_p[i][x] * m_qtild[i][y];
           }

    }


    Eigen::Matrix3d S = apq.transpose() * apq ; //symmetric matrix

    //3. Jacobi Diagonalisation
    Eigen::EigenSolver<Eigen::Matrix3d> es(S);

    //V * D * V^(-1)
    Eigen::Matrix3d D = es.pseudoEigenvalueMatrix();
    Eigen::Matrix3d U = es.pseudoEigenvectors() ;

    for(int j = 0; j < 3; j++)
    {
        if(D(j,j) <= 0)
        {
            D(j,j) = 0.05f;
        }
        D(j,j) = 1.0f/sqrt(D(j,j));
    }

    S = U * D * U.transpose();

    Eigen::Matrix3d R = apq * S; //S^{-1}


    // compute R~
    Matrix39d Rtild;
    Rtild.setZero(3,9);

    for(unsigned int x=0; x<3 ; ++x)
      for(unsigned int y=0; y<3 ; ++y)
          Rtild(x,y) = R(x,y);


       // compute quadratic deformation
    Matrix39d Atild = Apqtild * m_aqqtild;

    REAL det = Atild.determinant();
    det = 1.0f/powf(fabs(det),1.0f/9.0f);

    if(det<0.0f)
        det = -det;

    // \beta * A~ + (1 - \beta) * R~
    Matrix39d T = this->m_beta * Atild * det + (1.0f - this->m_beta) * Rtild;

    //5.
    for(unsigned int i = this->m_goal.begin() ; i < this->m_goal.end() ; this->m_goal.next(i))
    {

        Eigen::Vector3d tmp = T * this->m_qtild[i];
        tmp += xcm; // g_{i} = T * q_i + x_{cm}

        VEC3 g;
        for (unsigned int j = 0 ; j < 3 ; ++j)
             g[j] = tmp(j);

         this->m_goal[i] = g;
    }

}

} // namespace ShapeMatching

} // namespace Simulation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
