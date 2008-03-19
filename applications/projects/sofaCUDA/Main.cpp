#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#include <sofa/gpu/cuda/mycuda.h>

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

using namespace sofa::simulation::tree;
using namespace sofa::gpu::cuda;

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------

//#define EXECUTION 1

#ifndef EXECUTION

int main(int argc, char** argv)
{
    //std::string fileName = "beam10x10x46-spring-rk4-CUDA.scn";
    std::string fileName = "quadSpringSphereCUDA.scn";

    int nbIter = 0;
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" filename.scn [niterations]\n";
        //return 1;
    }
    else
    {
        fileName = argv[1];
        if (argc >=3) nbIter = atoi(argv[2]);
    }

    sofa::gui::SofaGUI::Init(argv[0]);

    sofa::helper::system::DataRepository.findFile(fileName);

    mycudaInit(0);

    GNode* groot = NULL;
    ctime_t t0, t1;
    CTime::getRefTime();

    if (!fileName.empty())
    {
        groot = getSimulation()->load(fileName.c_str());
    }

    if (groot==NULL)
    {
        groot = new GNode;
    }

    if (nbIter > 0)
    {

        groot->setAnimate(true);

        std::cout << "Computing first iteration." << std::endl;

        getSimulation()->animate(groot);

        //=======================================
        // Run the main loop

        std::cout << "Computing " << nbIter << " iterations." << std::endl;
        t0 = CTime::getRefTime();

        //=======================================
        // SEQUENTIAL MODE
        int n = 0;
        for (int i=0; i<nbIter; i++)
        {
            int n2 = i*80/(nbIter-1);
            while(n2>n)
            {
                std::cout << '.' << std::flush;
                ++n;
            }
            getSimulation()->animate(groot);
        }

        t1 = CTime::getRefTime();
        std::cout << std::endl;
        std::cout << nbIter << " iterations done." << std::endl;
        std::cout << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it." << std::endl;
        std::string logname = fileName.substr(0,fileName.length()-4)+"-log.txt";
        std::ofstream flog(logname.c_str());
        flog << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it." << std::endl;
        flog.close();
        std::string objname = fileName.substr(0,fileName.length()-4)+"-scene.obj";
        std::cout << "Exporting to OBJ " << objname << std::endl;
        getSimulation()->exportOBJ(groot, objname.c_str());
        std::string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
        std::cout << "Exporting to XML " << xmlname << std::endl;
        getSimulation()->exportXML(groot, xmlname.c_str());

    }
    else
    {
        sofa::gui::SofaGUI::MainLoop(groot,fileName.c_str());
    }

    return 0;
}

#else

#include <sofa/gpu/cuda/CudaLCP.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/helper/system/thread/CTime.h>
using sofa::helper::system::thread::CTime;

static int sz = 2048;
static int dim;

static float ** read_m;
static float * read_q;
static float * read_f;

void readFile(const char * filename)
{
    FILE * f = fopen(filename,"r");
    if (f==NULL)
    {
        printf("fichier non trouvé\n");
        exit(1);
    }

    char buf[64];
    fgets(buf,64,f);

    sscanf(buf,"%d",&dim);
    printf("dim = %d\n",dim);
    read_m = (float**) malloc(sz*sizeof(float*));
    read_f = (float*) malloc(sz*sizeof(float));
    read_q  = (float*) malloc(sz*sizeof(float));
    for (int i=0; i<sz; i++) read_m[i] = (float*) malloc(sz*sizeof(float));

    fgets(buf,64,f);//on enléve la ligne de commentaire

    printf("%s",buf);
    fgets(buf,64,f);//on enléve la ligne de commentaire
    for (int i=0; i<dim; i++)
    {
        for (int j=0; j<dim; j++)
        {
            sscanf(buf,"%f",&read_m[i][j]);
            fgets(buf,64,f);
        }
    }

    printf("%s",buf);
    fgets(buf,64,f);
    for (int i=0; i<dim; i++)
    {
        sscanf(buf,"%f",&read_f[i]);
        fgets(buf,64,f);
    }

    printf("%s",buf);
    fgets(buf,64,f);
    for (int i=0; i<dim; i++)
    {
        sscanf(buf,"%f",&read_q[i]);
        fgets(buf,64,f);
    }

    for (int i=0; i<sz; i++)
    {
        for (int j=0; j<sz; j++) read_m[i][j] = read_m[i%dim][j%dim];
        read_f[i] = read_f[i%dim];
        read_q[i] = read_q[i%dim];
    }
}


#if EXECUTION == 1

int main(int argc, char** argv)
{
    int itMax = 10;
    float tol = -0.001;

    readFile("dumpLCP.dat");

    FILE * fc = fopen("courbe.csv","w");

    CudaMatrix<float> cuda_M;
    CudaVector<float> cuda_q;
    CudaVector<float> cuda_f0;
    CudaVector<float> cuda_f1;
    CudaVector<float> cuda_f2;
    CudaVector<float> cuda_f3;
    CudaVector<float> cuda_f4;
    CudaVector<float> cuda_f5;
    CudaVector<float> cuda_f6;
    CudaVector<float> cuda_r;

    cuda_M.resize(sz,sz,64);
    cuda_q.resize(sz);
    cuda_f0.resize(sz);
    cuda_f1.resize(sz);
    cuda_f2.resize(sz);
    cuda_f3.resize(sz);
    cuda_f4.resize(sz);
    cuda_f5.resize(sz);
    cuda_f6.resize(sz);
    cuda_r.resize(sz);
    double scale = 1.0 / (double)CTime::getRefTicksPerSec();

    CudaLCP::CudaGaussSeidelLCP1(6,sz,cuda_q,cuda_M,cuda_f6,cuda_r,tol,0);
    CudaLCP::CudaGaussSeidelLCP1(1,sz,cuda_q,cuda_M,cuda_f6,cuda_r,tol,0);

    for (int it=1; it<sz; it++)
    {
        cuda_M.resize(it,it,64);
        cuda_q.resize(it);
        cuda_f0.resize(it);
        cuda_f1.resize(it);
        cuda_f2.resize(it);
        cuda_f3.resize(it);
        cuda_f4.resize(it);
        cuda_f5.resize(it);
        cuda_f6.resize(it);
        cuda_r.resize(it);

        for (int i=0; i<it; i++)
        {
            for (int j=0; j<it; j++) cuda_M[i][j] = read_m[i][j];
            cuda_q[i] = read_q[i];
            cuda_f0[i] = read_f[i];
            cuda_f1[i] = read_f[i];
            cuda_f2[i] = read_f[i];
            cuda_f3[i] = read_f[i];
            cuda_f4[i] = read_f[i];
            cuda_f5[i] = read_f[i];
            cuda_f6[i] = read_f[i];
        }
        unsigned long long time0 = 0;
        unsigned long long time1 = 0;
        unsigned long long time2 = 0;
        unsigned long long time3 = 0;
        unsigned long long time4 = 0;
        unsigned long long time5 = 0;
        unsigned long long time6 = 0;

        cuda_M.hostWrite();
        cuda_q.hostWrite();
        time0 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(0,it,cuda_q,cuda_M,cuda_f0,cuda_r,tol,itMax);
        time0 = CTime::getRefTime() - time0;

        cuda_M.hostWrite();
        cuda_q.hostWrite();
        time1 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(1,it,cuda_q,cuda_M,cuda_f1,cuda_r,tol,itMax);
        cuda_r.deviceRead();
        time1 = CTime::getRefTime() - time1;

        cuda_M.hostWrite();
        cuda_q.hostWrite();
        time2 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(2,it,cuda_q,cuda_M,cuda_f2,cuda_r,tol,itMax);
        cuda_r.deviceRead();
        time2 = CTime::getRefTime() - time2;

        if (it<512)
        {
            cuda_M.hostWrite();
            cuda_q.hostWrite();
            time3 = CTime::getRefTime();
            CudaLCP::CudaGaussSeidelLCP1(3,it,cuda_q,cuda_M,cuda_f3,cuda_r,tol,itMax);
            cuda_r.deviceRead();
            time3 = CTime::getRefTime() - time3;
        }

        cuda_M.hostWrite();
        cuda_q.hostWrite();
        time4 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(4,it,cuda_q,cuda_M,cuda_f4,cuda_r,tol,itMax);
        cuda_r.deviceRead();
        time4 = CTime::getRefTime() - time4;

        cuda_M.hostWrite();
        cuda_q.hostWrite();
        time5 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(5,it,cuda_q,cuda_M,cuda_f5,cuda_r,tol,itMax);
        cuda_r.deviceRead();
        time5 = CTime::getRefTime() - time5;

        cuda_M.hostWrite();
        cuda_q.hostWrite();
        time6 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(6,it,cuda_q,cuda_M,cuda_f6,cuda_r,tol,itMax);
        cuda_r.deviceRead();
        time6 = CTime::getRefTime() - time6;

        double t0 = time0*scale;
        double t1 = time1*scale;
        double t2 = time2*scale;
        double t3 = time3*scale;
        double t4 = time4*scale;
        double t5 = time5*scale;
        double t6 = time6*scale;

        fprintf(fc,"%d %f %f %f %f %f %f %f\n",it,t0,t1,t2,t3,t4,t5,t6);

        std::cout << it << endl;
    }

    fclose(fc);

}
#endif

#if EXECUTION == 2
int main(int argc, char** argv)
{
    int itMax = 10;
    float tol = -0.001;

    readFile("dumpLCP.dat");

    FILE * fc = fopen("courbe2.csv","w");

    CudaMatrix<float> cuda_M;
    CudaVector<float> cuda_q;
    CudaVector<float> cuda_f;
    CudaVector<float> cuda_r;

    cuda_M.resize(sz,sz,64);
    cuda_q.resize(sz);
    cuda_f.resize(sz);
    cuda_r.resize(sz);
    double scale = 1.0 / (double)CTime::getRefTicksPerSec();

    for (int it=1; it<sz; it++)
    {
        cuda_M.resize(it,it,64);
        cuda_q.resize(it);
        cuda_f.resize(it);
        cuda_r.resize(it);

        for (int i=0; i<it; i++)
        {
            for (int j=0; j<it; j++) cuda_M[i][j] = read_m[i][j];
            cuda_q[i] = read_q[i];
            cuda_f[i] = read_f[i];
        }
        unsigned long long time0 = 0;
        unsigned long long time1 = 0;

        time0 = CTime::getRefTime();
        cuda_M.deviceRead();
        cuda_q.deviceRead();
        cuda_f.deviceRead();
        cuda_r.deviceWrite();
        time0 = CTime::getRefTime() - time0;

        time1 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(6,it,cuda_q,cuda_M,cuda_f,cuda_r,tol,itMax);
        cuda_r.deviceRead();
        time1 = CTime::getRefTime() - time1;

        double t0 = time0*scale;
        double t1 = time1*scale;

        fprintf(fc,"%d %f %f\n",it,t0,t1);

        std::cout << it << endl;
    }

    fclose(fc);
}
#endif

#if EXECUTION == 3
int main(int argc, char** argv)
{
    float tol = 0.001;
    int dim = 256;

    readFile("dumpLCP.dat");

    FILE * fc = fopen("courbe3.csv","w");

    CudaMatrix<float> cuda_M;
    CudaVector<float> cuda_q;
    CudaVector<float> cuda_f;
    CudaVector<float> cuda_r;

    cuda_M.resize(dim,dim,64);
    cuda_q.resize(dim);
    cuda_f.resize(dim);
    cuda_r.resize(dim);
    double scale = 1.0 / (double)CTime::getRefTicksPerSec();

    for (int i=0; i<dim; i++)
    {
        for (int j=0; j<dim; j++) cuda_M[i][j] = read_m[i][j];
        cuda_q[i] = read_q[i];
    }

    for (int it=1; it<sz; it++)
    {
        for (int i=0; i<dim; i++) cuda_f[i] = read_f[i];

        unsigned long long time0 = 0;
        unsigned long long time1 = 0;

        time0 = CTime::getRefTime();
        cuda_M.deviceRead();
        cuda_q.deviceRead();
        cuda_f.deviceRead();
        cuda_r.deviceWrite();
        time0 = CTime::getRefTime() - time0;

        time1 = CTime::getRefTime();
        CudaLCP::CudaGaussSeidelLCP1(6,dim,cuda_q,cuda_M,cuda_f,cuda_r,tol,it);
        cuda_r.deviceRead();
        time1 = CTime::getRefTime() - time1;

        double t0 = time0*scale;
        double t1 = time1*scale;

        fprintf(fc,"%d %f %f\n",it,t0,t1);

        std::cout << it << endl;
    }

    fclose(fc);
}
#endif

#endif
