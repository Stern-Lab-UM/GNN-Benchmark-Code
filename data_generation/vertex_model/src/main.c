//****************PARAMETERS************************
double kA;
double P0;
double Sig;
double Sig0;
double kM=1.;
double Time=0;
double time1=1000;
double time2=7000;
double time3=2000;
double h0=0.001;
double lth=0.01;
double h;
int packageID;
int SigI;
int T1EdgeID_1;
int T1EdgeID_2;
int Nx;
double shearFactor;
double dRmax=0;
double dGmax=0;

//*******************DECLARATIONS*************************
#include "_functions.h"

//********************MAIN********************************
int main(int argc, char *argv[]){


    //INPUT PARAMETERS
    if(argc != 8) { std::cerr << "Error! Wrong number of input elements!" << std::endl; return -1; }
    Nx= std::atoi(argv[1]); //even number greater or equal than 8
    kA = std::stod(argv[2]); //cell incompressibility modulus
    packageID = std::atoi(argv[3]); //essentialy determines the seed for random number generator
    SigI = std::atoi(argv[4]); Sig0=SigI*0.01; //from 0 to 50
    shearFactor = std::stod(argv[5]); //T1 edge 2
    T1EdgeID_1 = std::atoi(argv[6]); //T1 edge 1
    T1EdgeID_2 = std::atoi(argv[7]); //T1 edge 2
    int rndSeed=(packageID-1)*51+SigI+1; srand(rndSeed);
    array_max=100000;
    allocate();
    printf("Nc=%d\tkA=%g\tpackageID=%d\tSigI=%d\tshearFactor=%g\n", (int)(Nx*Nx), kA, packageID, SigI, shearFactor);
    //*******************

    //SAMPLE PREPARATION
    if(T1EdgeID_1<=0 && T1EdgeID_2<=0){
        printf("preparing sample...\n");
        //INITIALIZE
        set_initial_regHex(Nx);
        //STEP 1
        Time=0; Sig=0.5;
        while(Time<=time1){ eqOfMotion(); T1_transitions(0.001,0.02,12); }
        printf("step 1 done\n");
        //*******************
        //STEP 2
        Time=0; Sig=Sig0;
        while(Time<=time2){ eqOfMotion(); T1_transitions(0.001,0.02,12); }
        T1_transitions(0.001,-1,2);
        printf("step 2 done\n");
        //*******************
        //STEP 3
        Time=0; Sig=0;
        while(Time<=time3) eqOfMotion();
        printf("step 3 done\n");
        //*******************
        //OUT
        out_Vertissue2D();
    }

    else{
        //INITIALIZE
        set_initial_fromFile();
        expand_box(shearFactor*perioXYZ[1], perioXYZ[2]/shearFactor);
        Time=0; Sig=0;
        while(Time<=time3) eqOfMotion();
        for(int i=1; i<=Ne; i++) if(exist[2][i]==1) e_length_initial[i]=edge_length(i);
        //*******************
        //PERFORM T1
        if(T1EdgeID_1>0) printf("T1 on edge %d\n", T1EdgeID_1);
        if(T1EdgeID_1>0) single_T1(T1EdgeID_1);
        if(T1EdgeID_2>0) printf("T1 on edge %d\n", T1EdgeID_2);
        if(T1EdgeID_2>0) single_T1(T1EdgeID_2);
        //*******************
        //RELAX AFTER T1
        Time=0; Sig=0;
        while(Time<=time3) eqOfMotion();
        //*******************
        //OUT
        outGraph();
    }

    //DEALLOCATE
    deallocate();
    //*************************


    return 0;
}
//********************************************************
