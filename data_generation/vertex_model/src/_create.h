//****************************************************************************
//****************************************************************************
//******************************CREATE****************************************
//****************************************************************************
//****************************************************************************
int make_vertex(double x, double y){

    //ID
    int i=getID(1);

    //CREATE
    exist[1][i]=1;
    /* v */         v[i][0]=i; v[i][1]=x; v[i][2]=y; torus_vertex(i);
    /* v_edges */   v_edges[i][0]=0;
    /* v_cells */   v_cells[i][0]=0;

    //ATTRIBUTES
    for(int j=1; j<=2; j++) v_F[i][j]=0;
    /* v_resolveCells */   v_resolveCells[i][0]=0;
    /* v_clock */          v_clock[i]=0;
    /* v_T1edge */         v_T1edge[i]=0;

    return i;
}
//****************************************************************************
int make_edge(int v1, int v2, int i){

    //ID
    if(i==0) i=getID(2);

    //CREATE
    exist[2][i]=1;
    /* e */         e[i][0]=i; e[i][1]=v1; e[i][2]=v2;
    /* e_cells */   e_cells[i][0]=0;
    /* e_Vcells */  e_Vcells[i][0]=0;

    //EXTERNAL ATTRIBUTES
    /* v_edges */   for(int j=1; j<=2; j++) appendToArray(v_edges,e[i][j],i);

    //ATTRIBUTES
    if(i==0) reset_edge_attributes(i);

    return i;
}
//****************************************************************************
void remake_edge(int i, int vPrev, int vNew)
{
    int vv1=e[i][1]; if(vv1==vPrev) vv1=vNew;
    int vv2=e[i][2]; if(vv2==vPrev) vv2=vNew;

    dissolve_edge(i,1);
    make_edge(vv1,vv2,i);
}
//****************************************************************************
int cell_sides(int nrSides, int *sides, int i){

    //ID
    if(i==0) i=getID(3);

    //CREATE
    /* c */         c[i][0]=0; for(int j=1; j<=nrSides; j++) appendToArray(c,i,sides[j]);

    return i;
}
//****************************************************************************
void make_cell(int i){

    //CREATE
    exist[3][i]=1;
    /* c_vertices */c_vertices[i][0]=0;
    for(int j=1; j<=c[i][0]; j++){
        int eID=abs(c[i][j]);
        int v1=e[eID][1];
        int v2=e[eID][2];
        if(c[i][j]>0) appendToArray(c_vertices,i,v1);
        if(c[i][j]<0) appendToArray(c_vertices,i,v2);
    }
    /* c_Vedges */  c_Vedges[i][0]=0;
    for(int j=1; j<=c_vertices[i][0]; j++){
        int vID=c_vertices[i][j];
        for(int k=1; k<=v_edges[vID][0]; k++){
            int eID=v_edges[vID][k];
            if( isItin(c,i,eID)==0 ) appendToArray(c_Vedges,i,eID);
        }
    }

    //EXTERNAL ATTRIBUTES
    /* v_cells */   for(int j=1; j<=c_vertices[i][0]; j++)  appendToArray(v_cells,c_vertices[i][j],i);
    /* e_cells */   for(int j=1; j<=c[i][0]; j++)           appendToArray(e_cells,abs(c[i][j]),i);
    /* e_Vcells*/   for(int j=1; j<=c_Vedges[i][0]; j++)    appendToArray(e_Vcells,c_Vedges[i][j],i);


    //ATTRIBUTES
    c_area[i]=0;
    c_A0[i]=0;
}
//****************************************************************************
