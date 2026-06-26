//****************************************************************************
//****************************************************************************
//***********************EQUATION OF MOTION***********************************
//****************************************************************************
//****************************************************************************
double calc_forces(){

    //EDGE LENGTHS
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1) e_length[i]=edge_length(i);

    //CELL AREAS
    for(int i=1; i<=Nc; i++) if(exist[3][i]==1) c_area[i]=CellArea_new(i);

    //POTENTIAL-ENERGY CONTRIBUTIONS
    wA=0; wP=0;
    for(int i=1; i<=Nc; i++) if(exist[3][i]==1){
        wA += c_AreaCompressibility_force_New(i);
    }

    //LENGTH FORCE CONTRIBUTION
    wL=0;for(int i=1; i<=Ne; i++) if(exist[2][i]==1) wL+=e_spring_force(i);

    //MAX FORCE
    double Fmax=0;
    for(int i = 1; i <= Nv; i++) if(exist[1][i]==1){
        double foRce=sqrt(v_F[i][1]*v_F[i][1]+v_F[i][2]*v_F[i][2]);
        if(foRce>Fmax) Fmax=foRce;
    }
    return Fmax;
}
//****************************************************************************
void propagate_tension(){
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1){
        e_g[i] +=  -h*kM*e_g[i] + sqrt(2.*Sig*Sig*h*kM)*GaussianVariate();
    }
}
//****************************************************************************
void reset_forces(){
    for(int i=1; i<=Nv; i++) if(exist[1][i]==1) for(int j=1; j<=2; j++) v_F[i][j] = 0;
}
//****************************************************************************
double eqOfMotion(){

    //CALCULATE FORCES
    calc_forces();

    //TIME STEP
    h=h0;

    //PROPAGATE VERTICES
    for(int i=1; i<=Nv; i++) if(exist[1][i]==1){
        v[i][1] += h*v_F[i][1];
        v[i][2] += h*v_F[i][2];
        torus_vertex(i);
    }

    //PROPAGATE TENSIONS
    propagate_tension();

    //RESETS FORCES
    reset_forces();

    //UPDATE EDGE LENGTHS
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1) {
        double elen=edge_length(i);
        e_dl[i]=(elen-e_length[i]);
        e_length[i]=elen;
    }

    //UPDATE CELL AREAS
    for(int i=1; i<=Nc; i++) if(exist[3][i]==1) c_area[i]=CellArea_new(i);


    //t=t+dt
    printf("Time=%g\t\tP=%.16g\n", Time, wL);

    Time+=h;

    return wA+wP;
}
//****************************************************************************
