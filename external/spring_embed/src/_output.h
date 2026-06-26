//****************************************************************************
//****************************************************************************
//******************************OUTPUT****************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
double out_polygons(int fcount){


    char filename2[100]; char filename3[100];
    snprintf(filename2, sizeof(char) * 100, "./output/X_%d.txt", fcount);
    snprintf(filename3, sizeof(char) * 100, "./output/Y_%d.txt", fcount);

    double *dxdydz = new double[3];
    dxdydz[1]=0; dxdydz[2]=0;
    int vert_ref_id, vert_id;
    FILE *file2; file2 = fopen(filename2, "wt");
    FILE *file3; file3 = fopen(filename3, "wt");

    for(int i=1; i<=Nc; i++) if(exist[3][i]==1){

        int srgn=1;

        fprintf(file2, "%d ", srgn*c_vertices[i][0]);
        fprintf(file3, "%d ", srgn*c_vertices[i][0]);

        vert_ref_id=c_vertices[i][1];
        fprintf(file2, "%.16g ", v[vert_ref_id][1]);
        fprintf(file3, "%.16g ", v[vert_ref_id][2]);

        for(int j = 2; j <= c_vertices[i][0]; j++){
            vert_id=c_vertices[i][j];
            torus_dx_dy_dz(dxdydz,vert_id,vert_ref_id);
            fprintf(file2, "%.16g ", v[vert_id][1] + dxdydz[1]);
            fprintf(file3, "%.16g ", v[vert_id][2] + dxdydz[2]);
        }

        for(int j = c_vertices[i][0]; j <= 15; j++){
            fprintf(file2, "0 ");
            fprintf(file3, "0 ");
        }

        fprintf(file2, "\n");
        fprintf(file3, "\n");
    }
    fclose(file2);
    fclose(file3);

    delete []dxdydz;

    return 0;
}
//****************************************************************************
void out_tissue(int fcount){
    out_polygons(fcount);
}
//****************************************************************************
void out_Vertissue2D(){

    char filename2[100];
    snprintf(filename2, sizeof(char) * 100, "./output/final_%g.vt2d", P0);
    FILE *file1; file1 = fopen(filename2, "wt");

    int nrV=0, nrE=0, nrC=0;

    //ALLOCATE INTERNAL VARIABLES
    int **v_natural_id; int **e_natural_id; int **c_natural_id;
    v_natural_id  = new int*[Nv+1]; e_natural_id  = new int*[Ne+1]; c_natural_id  = new int*[Nc+1];
    for(int i=0; i < Nv+1; i++) v_natural_id[i]  = new int[3];
    for(int i=0; i < Ne+1; i++) e_natural_id[i]  = new int[3];
    for(int i=0; i < Nc+1; i++) c_natural_id[i]  = new int[3];

    //RESET NATURAL IDS
    for(int i=1; i<=Nv; i++){
        v_natural_id[i][1]=0;
        v_natural_id[i][2]=0;
    }
    for(int i=1; i<=Ne; i++){
        e_natural_id[i][1]=0;
        e_natural_id[i][2]=0;
    }
    for(int i=1; i<=Nc; i++){
        c_natural_id[i][1]=0;
        c_natural_id[i][2]=0;
    }

    //VERTICES
    int cnt=0;
    for(int i=1; i<=Nv; i++) if(exist[1][i]==1){
        cnt++;
        v_natural_id[i][1]=cnt;
        v_natural_id[cnt][2]=i;
    }
    nrV=cnt;

    //EDGES
    cnt=0;
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1){
        cnt++;
        e_natural_id[i][1]=cnt;
        e_natural_id[cnt][2]=i;
    }
    nrE=cnt;

    //CELLS
    cnt=0;
    for(int i=1; i<=Nc; i++) if(exist[3][i]==1){
        cnt++;
        c_natural_id[i][1]=cnt;
        c_natural_id[cnt][2]=i;
    }
    nrC=cnt;
    fprintf(file1, "%d  %d  %d\n\n", nrV, nrE, nrC);
    fprintf(file1, "%g  %g\n\n", perioXYZ[1], perioXYZ[2]);


    //VERTICES
    for(int i=1; i<=Nv; i++) if(exist[1][i]==1) fprintf(file1, "%f  %f\n", v[i][1], v[i][2]);
    fprintf(file1, "\n");

    //EDGES
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1) fprintf(file1, "%d  %d  %g\n", v_natural_id[e[i][1]][1], v_natural_id[e[i][2]][1], e_l0[i]);
    fprintf(file1, "\n");

    //CELLS
    for(int i=1; i<=Nc; i++) if(exist[3][i]==1){
        fprintf(file1, "%d  ", c[i][0]);
        for(int j=1; j<=c[i][0]; j++){
            if(c[i][j]<0) fprintf(file1, "%d  ", -e_natural_id[abs(c[i][j])][1]);
            else fprintf(file1, "%d  ", e_natural_id[abs(c[i][j])][1]);
        }
        for(int j=15; j>c[i][0]; j--){
            fprintf(file1, "0  ");
        }
        fprintf(file1, "\n");
    }

    //DEALLOCATE INTERNAL VARIABLES
    for(int i=0; i < Nv+1; ++i) delete [] v_natural_id[i] ;
    for(int i=0; i < Ne+1; ++i) delete [] e_natural_id[i] ;
    for(int i=0; i < Nc+1; ++i) delete [] c_natural_id[i] ;
    delete [] v_natural_id;
    delete [] e_natural_id;
    delete [] c_natural_id;

    fclose(file1);
}
//****************************************************************************
