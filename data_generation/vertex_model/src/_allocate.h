//****************************************************************************
//****************************************************************************
//******************************ALLOCATE**************************************
//****************************************************************************
//****************************************************************************
void reset_arrays(){

    Nv = 0; Ne = 0; Nc = 0;


    //ids
    for(int i = 1; i<=3; i++) ids[i][0]=0;


    for(int i = 1; i<=array_max; i++){

        //exist
        for(int j=1; j<=3; j++) exist[j][i]=0;

        //vertices
        for(int j = 0; j <= 2; j++) v[i][j]=0;
        v_edges[i][0]=0;
        v_cells[i][0]=0;
        v_resolveCells[i][0]=0;
        v_T1edge[i]=0;
        v_fixed[i]=0;
        //.................
        for(int j = 0; j <= 2; j++) v_F[i][j]=0;
        //.................
        v_clock[i]=0;

        //edges
        for(int j = 0; j <= 2; j++) e[i][j]=0;
        e_cells[i][0]=0;
        e_Vcells[i][0]=0;
        //.................
        e_g[i]=0;
        e_length[i]=0;
        e_length_initial[i]=0;
        e_dl[i]=0;

        //cells
        c[i][0]=0;
        c_vertices[i][0]=0;
        c_Vedges[i][0]=0;
        c_area[i]=0;
        c_A0[i]=0;
        c_type[i]=0;
    }
}
//****************************************************************************
void allocate(){

    //ids
    ids     = new int*[4];
    for(int i=0; i <= 3; i++) ids[i]   = new int[array_max+1];


    //exist
    exist     = new int*[4];
    for(int i=0; i <= 3; i++) exist[i] = new int[array_max+1];


    //vertices
    v          = new double*[array_max+1];
    v_edges   = new int*[array_max+1];
    v_cells   = new int*[array_max+1];
    v_resolveCells   = new int*[array_max+1];
    //.................
    v_F        = new double*[array_max+1];
    for(int i=0; i <= array_max; i++){
        v[i]          = new double[3];
        v_edges[i]   = new int[200];
        v_cells[i]   = new int[200];
        v_resolveCells[i]   = new int[200];
        //.................
        v_F[i]          = new double[3];
    }
    v_clock = new double[array_max+1];
    v_T1edge = new int[array_max+1];
    v_fixed = new int[array_max+1];


    //edges
    e          = new int*[array_max+1];
    e_cells   = new int*[array_max+1];
    e_Vcells   = new int*[array_max+1];
    for(int i=0; i <= array_max; i++){
        e[i]          = new int[3];
        e_cells[i]   = new int[200];
        e_Vcells[i]   = new int[200];
    }
    e_g = new double[array_max+1];
    e_length = new double[array_max+1];
    e_length_initial = new double[array_max+1];
    e_dl = new double[array_max+1];

    //cells
    c   = new int*[array_max+1];
    c_vertices   = new int*[array_max+1];
    c_Vedges   = new int*[array_max+1];
    for(int i=0; i <= array_max; i++){
        c[i]   = new int[200];
        c_vertices[i]   = new int[200];
        c_Vedges[i]   = new int[200];
    }
    c_area = new double[array_max+1];
    c_A0 = new double[array_max+1];
    c_type = new int[array_max+1];


    //MISC
    perioXYZ = new double[3];


    //RESET ARRAYS
    reset_arrays();

    return ;
}
//****************************************************************************
void deallocate(){

    //ids
    for(int i=0; i <= 3; i++) delete [] ids[i];
    delete [] ids;

    //exist
    for(int i=0; i <= 3; i++) delete [] exist[i];
    delete [] exist;


    //vertices
    for(int i=0; i <= array_max; i++){
        delete [] v[i]      ;
        delete [] v_edges[i]  ;
        delete [] v_cells[i]  ;
        delete [] v_resolveCells[i]  ;
        //.................
        delete [] v_F[i]      ;
    }
    delete [] v;
    delete [] v_edges  ;
    delete [] v_cells  ;
    delete [] v_resolveCells  ;
    //.................
    delete [] v_F;
    //.................
    delete [] v_clock;
    delete [] v_T1edge;
    delete [] v_fixed;


    //edges
    for(int i=0; i <= array_max; i++){
        delete [] e[i]      ;
        delete [] e_cells[i]  ;
        delete [] e_Vcells[i]  ;
    }
    delete [] e;
    delete [] e_cells  ;
    delete [] e_Vcells  ;
    //.................
    delete [] e_g  ;
    delete [] e_length  ;
    delete [] e_length_initial  ;
    delete [] e_dl  ;

    //cells
    for(int i=0; i <= array_max; i++){
        delete [] c[i]  ;
        delete [] c_vertices[i]      ;
        delete [] c_Vedges[i]  ;
    }
    delete [] c  ;
    delete [] c_vertices;
    delete [] c_Vedges  ;
    delete [] c_area  ;
    delete [] c_A0  ;
    delete [] c_type  ;

    //MISC
    delete [] perioXYZ;

    return ;
}
//****************************************************************************
