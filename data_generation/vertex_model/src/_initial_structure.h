//****************************************************************************
//****************************************************************************
//***********************INITIAL STRUCTURE************************************
//****************************************************************************
//****************************************************************************
void set_initial_regHex(int _Nx){

    int *sides; sides = new int[13];

    //INTERNAL VARIABLES
    double *v_stitching_edge;//only useful for building regular hexagonal network
    v_stitching_edge = new double[array_max];
    int Ny=_Nx/2;
    double dd=1.2408064788027995;
    double ddx=0.08;

    //GLOBAL VARIABLES
    perioXYZ[1] = _Nx*sqrt(3)*dd/2.;
    perioXYZ[2] = Ny*(dd+dd/2);

    for(int i=1; i<=Ny; i++){
        for (int j =1; j<=_Nx; j++){
            double x0=(j-1)*sqrt(3)*dd/2+ddx;
            double y0=(3*dd/2)*(i-1)+ddx;
            make_vertex(
                        x0,
                        y0-dd/2+dd/2
                        );
            make_vertex(
                        x0+sqrt(3)*dd/4,
                        y0-dd/4+dd/2
                        );
            make_vertex(
                        x0+sqrt(3)*dd/4,
                        y0+dd/4+dd/2
                        );
            make_vertex(
                        x0,
                        y0+dd/2+dd/2
                        );
        }

        make_edge(i*_Nx*4-2, (i-1)*_Nx*4+1,0);
        make_edge((i-1)*_Nx*4+1, (i-1)*_Nx*4+2,0);
        make_edge((i-1)*_Nx*4+2, (i-1)*_Nx*4+3,0);
        make_edge((i-1)*_Nx*4+3, (i-1)*_Nx*4+4,0);
        make_edge((i-1)*_Nx*4+4, i*_Nx*4-1,0);

        for (int j =2; j<=_Nx; j++){
            make_edge((i-1)*_Nx*4+(j-2)*4+2, (i-1)*_Nx*4+(j-2)*4+5,0);
            make_edge((i-1)*_Nx*4+(j-2)*4+5, (i-1)*_Nx*4+(j-2)*4+6,0);
            make_edge((i-1)*_Nx*4+(j-2)*4+6, (i-1)*_Nx*4+(j-2)*4+7,0);
            make_edge((i-1)*_Nx*4+(j-2)*4+7, (i-1)*_Nx*4+(j-2)*4+8,0);
            make_edge((i-1)*_Nx*4+(j-2)*4+8, (i-1)*_Nx*4+(j-2)*4+3,0);
        }
    }

    for (int i =1; i<=Ny-1; i++){
        for (int j =1; j<=_Nx; j++){
            int edgid=make_edge(_Nx*4*(i-1)+4+(j-1)*4, _Nx*4*i+1+(j-1)*4,0);
            v_stitching_edge[_Nx*4*(i-1)+4+(j-1)*4]=edgid;
        }
    }

    int i = Ny;
    for (int j=1; j<=_Nx; j++){
        int edgid=make_edge(_Nx*4*(i-1)+4+(j-1)*4, 1+(j-1)*4,0);
        v_stitching_edge[_Nx*4*(i-1)+4+(j-1)*4]=edgid;
    }



    //*************************************
    //*************************************
    //*************************************
    //BASAL EDGES
    //*************************************x
    //LIHE VRSTICE
    for (int i=1; i<=Ny; i++){

        for(int i=0; i<=12; i++) sides[i]=0;
        sides[1]=(i-1)*_Nx*5+1;
        sides[2]=(i-1)*_Nx*5+2;
        sides[3]=(i-1)*_Nx*5+3;
        sides[4]=(i-1)*_Nx*5+4;
        sides[5]=(i-1)*_Nx*5+5;
        sides[6]=-((i-1)*_Nx*5+(_Nx*5-2));
        sides[7]=0;
        sides[8]=0;
        sides[9]=0;
        sides[10]=0;
        sides[11]=0;
        sides[12]=0;
        cell_sides(6,sides,0);

        for (int j=6; j<=_Nx*5-4; j+=5){
            for(int i=0; i<=12; i++) sides[i]=0;
            sides[1]=(i-1)*_Nx*5+j;
            sides[2]=(i-1)*_Nx*5+j+1;
            sides[3]=(i-1)*_Nx*5+j+2;
            sides[4]=(i-1)*_Nx*5+j+3;
            sides[5]=(i-1)*_Nx*5+j+4;
            sides[6]=-((i-1)*_Nx*5+j-3);
            sides[7]=0;
            sides[8]=0;
            sides[9]=0;
            sides[10]=0;
            sides[11]=0;
            sides[12]=0;
            cell_sides(6,sides,0);
        }
    }

    //SODE VRSTICE
    for (int i=1; i<=Ny-1; i++){

        for (int j=1; j<=_Nx-1; j++){
            for(int i=0; i<=12; i++) sides[i]=0;
            sides[1]=-(4+(j-1)*5+(i-1)*_Nx*5);
            sides[2]=-(10+(j-1)*5+(i-1)*_Nx*5);
            sides[3]=v_stitching_edge[e[(10+(j-1)*5)+(i-1)*_Nx*5][1]];
            sides[4]=-(5*_Nx+6+(j-1)*5+(i-1)*_Nx*5);
            sides[5]=-(5*_Nx+6+(j-1)*5-4+(i-1)*_Nx*5);
            sides[6]=-(v_stitching_edge[e[(10+(j-1)*5)+(i-1)*_Nx*5][1]]-1);
            sides[7]=0;
            sides[8]=0;
            sides[9]=0;
            sides[10]=0;
            sides[11]=0;
            sides[12]=0;
            cell_sides(6,sides,0);
        }

        for(int i=0; i<=12; i++) sides[i]=0;
        sides[1]=-(4+(_Nx-1)*5+(i-1)*_Nx*5);
        sides[2]=-(5+(i-1)*_Nx*5);
        sides[3]=v_stitching_edge[e[10+(i-1)*_Nx*5][1]]-1;
        sides[4]=-(5*_Nx+6-4-1+(i-1)*_Nx*5);
        sides[5]=-(5*_Nx+6+(_Nx-1)*5-4+(i-1)*_Nx*5);
        sides[6]=-(v_stitching_edge[e[(10+(_Nx-1-1)*5+(i-1)*_Nx*5)][1]]);
        sides[7]=0;
        sides[8]=0;
        sides[9]=0;
        sides[10]=0;
        sides[11]=0;
        sides[12]=0;
        cell_sides(6,sides,0);
    }

    //ZADNJA VRSTICA
    for (int j=1; j<=_Nx-1; j++){

        for(int i=0; i<=12; i++) sides[i]=0;
        sides[1]=-(4+(j-1)*5+(Ny-1)*_Nx*5);
        sides[2]=-(10+(j-1)*5+(Ny-1)*_Nx*5);
        sides[3]=v_stitching_edge[e[(10+(j-1)*5)+(Ny-1)*_Nx*5][1]];
        sides[4]=-(6+(j-1)*5);
        sides[5]=-(2+(j-1)*5);
        sides[6]=-(v_stitching_edge[e[(10+(j-1)*5)+(Ny-1)*_Nx*5][1]]-1);
        sides[7]=0;
        sides[8]=0;
        sides[9]=0;
        sides[10]=0;
        sides[11]=0;
        sides[12]=0;
        cell_sides(6,sides,0);
    }

    for(int i=0; i<=12; i++) sides[i]=0;
    sides[1]=-(4+(_Nx-1)*5+(Ny-1)*_Nx*5);
    sides[2]=-(5+(Ny-1)*_Nx*5);
    sides[3]=v_stitching_edge[e[10+(Ny-1)*_Nx*5][1]]-1;
    sides[4]=-1;
    sides[5]=-(_Nx*5-3);
    sides[6]=-(v_stitching_edge[e[(10+(_Nx-1-1)*5+(Ny-1)*_Nx*5)][1]]);
    sides[7]=0;
    sides[8]=0;
    sides[9]=0;
    sides[10]=0;
    sides[11]=0;
    sides[12]=0;
    cell_sides(6,sides,0);


    delete [] v_stitching_edge;

    //MAKE CELLS
    for(int i=1; i<=Nc; i++) if(c[i][0]!=0) make_cell(i);


    delete [] sides;

    expand_box(sqrt(Nc*1.),sqrt(Nc*1.));
}
//****************************************************************************
void set_initial_fromFile(){

    int *sides; sides = new int[16];
    int nrSides, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;

    //FILE
    char filename2[100];
    snprintf(filename2, sizeof(filename2),"./output/final_%d_%g_%d_%d.vt2d",(int)(Nx*Nx), kA, packageID, SigI);
    printf("loading initial sample from %s\n", filename2);
    FILE *file1 = fopen(filename2, "rt");


    //V,E,P,C
    int nrV=0, nrE=0, nrC=0;
    fscanf(file1, "%d  %d  %d\n", &nrV, &nrE, &nrC);

    //perioXYZ
    fscanf(file1, "%lf  %lf\n", &perioXYZ[1], &perioXYZ[2]);

    //VERTICES
    double xx, yy, blank1, blank2;
    for(int i=1; i<=nrV; i++){
        fscanf(file1, "%lf  %lf\n", &xx, &yy);
        make_vertex(xx,yy);
    }

    //EDGES
    int v1, v2;
    double blank3;
    for(int i=1; i<=nrE; i++){
        fscanf(file1, "%d  %d\n", &v1, &v2);
        make_edge(v1,v2,0);
    }

    //POLYGONS
    for(int i=0; i<=15; i++) sides[i]=0;
    for(int i=1; i<=nrC; i++){
        fscanf(file1, "%d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d  %d\n", &nrSides, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        if(b1!=0) sides[1]=b1;
        if(b2!=0) sides[2]=b2;
        if(b3!=0) sides[3]=b3;
        if(b4!=0) sides[4]=b4;
        if(b5!=0) sides[5]=b5;
        if(b6!=0) sides[6]=b6;
        if(b7!=0) sides[7]=b7;
        if(b8!=0) sides[8]=b8;
        if(b9!=0) sides[9]=b9;
        if(b10!=0) sides[10]=b10;
        if(b11!=0) sides[11]=b11;
        if(b12!=0) sides[12]=b12;
        if(b13!=0) sides[13]=b13;
        if(b14!=0) sides[14]=b14;
        if(b15!=0) sides[15]=b15;
        cell_sides(nrSides,sides,0);
    }
    delete [] sides;

    //MAKE CELLS
    for(int i=1; i<=Nc; i++) if(c[i][0]!=0) make_cell(i);

    fclose(file1);
}
//****************************************************************************
