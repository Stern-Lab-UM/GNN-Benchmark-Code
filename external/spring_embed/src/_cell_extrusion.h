//****************************************************************************
int reduce_edge_number(int i)
{

    for(int j=1; j<=c[i][0]; j++){
        int eID=abs(c[i][j]);
        int vID=T1_EDGE_TO_VERTEX(eID);
        v_clock[vID]=1;
        if( v_clock[vID]>=0) {
            T1_VERTEX_TO_EDGE(vID,0.01);
            v_clock[vID]=0;
        }
        if(c[i][0]==3) break;
    }

    return 0;
}
//****************************************************************************
int vert_edg_cells_ce(int i, int *vertices, int *edges, int *cells)
{
    //VERTICES
    for(int j=1; j<=c_vertices[i][0]; j++){
        appendToList(vertices,c_vertices[i][j]);
    }

    //CELLS
    appendToList(cells,i);
    for(int j=1; j<=c[i][0]; j++){
        int cID1=e_cells[abs(c[i][j])][1];
        int cID2=e_cells[abs(c[i][j])][2];
        if(cID1!=i) appendToList(cells,cID1);
        else if(cID2!=i) appendToList(cells,cID2);
    }

    //EDGES
    for(int j=1; j<=c[i][0]; j++){
        int eID=abs(c[i][j]);
        appendToList(edges,eID);
    }
    for(int j=1; j<=c_vertices[i][0]; j++){
        int vID=c_vertices[i][j];
        for(int k=1; k<=v_edges[vID][0]; k++){
            int eID=v_edges[vID][k];
            if(isItinList(edges,eID)==0) appendToList(edges,eID);
        }
    }

    for(int j=1; j<=edges[0]; j++){
        int eID=edges[j];
        for(int k=1; k<=e_Vcells[eID][0]; k++) appendToList(cells,e_Vcells[eID][k]);
    }

    return 0;
}
//****************************************************************************
int collapseVertices(int i)
{
    double *dxdydz = new double[3];
    dxdydz[1]=0; dxdydz[2]=0;
    int vert_ref_id, vert_id;

    double sumX=0, sumY=0;

    vert_ref_id=c_vertices[i][1];
    sumX+=v[vert_ref_id][1];
    sumY+=v[vert_ref_id][2];

    for(int j = 2; j <= c_vertices[i][0]; j++){
        vert_id=c_vertices[i][j];
        torus_dx_dy_dz(dxdydz,vert_id,vert_ref_id);
        sumX+=v[vert_id][1] + dxdydz[1];
        sumY+=v[vert_id][2] + dxdydz[2];
    }

    sumX/=c_vertices[i][0];
    sumY/=c_vertices[i][0];

    delete []dxdydz;

    for(int j = 2; j <= c_vertices[i][0]; j++){
        vert_id=c_vertices[i][j];
        v[vert_id][1]=sumX;
        v[vert_id][2]=sumY;
        torus_vertex(vert_id);
    }

    return 0;
}
//****************************************************************************
int CELL_EXTRUSION(int i)
{

//    reduce_edge_number(i);
//    if(c[i][0]!=3) return 0;

    //******************************************************************************
    int *vertices; vertices = new int[200]; for(int j=0; j<=199; j++) vertices[j]=0;
    int *edges; edges = new int[200]; for(int j=0; j<=199; j++) edges[j]=0;
    int *cells; cells = new int[200]; for(int j=0; j<=199; j++) cells[j]=0;
    vert_edg_cells_ce(i,vertices,edges,cells);
//    printf("vertices\n"); printLIST(vertices);
//    printf("edges\n"); printLIST(edges);
//    printf("cells\n"); printLIST(cells);
    //******************************************************************************


    //COLLAPSE CELL VERTICES
    collapseVertices(i);


    //DISSOLVE CELLS
    for(int j=1; j<=cells[0]; j++) dissolve_cell(cells[j]);
    delete_cell(i);


    //REMOVE EDGEs FROM LIST OF CELLS
    for(int j=1; j<=3; j++){
        int eID=edges[j];
        for(int k=1; k<=cells[0]; k++){
            int cID=cells[k];
            if(isItin(c,cID,eID)!=0) removeELEMENT(c,cID,eID);
        }
        dissolve_edge(eID,0);
    }


    //RESTITCH EDGES
    for(int j=5; j<=edges[0]; j++){
        remake_edge(edges[j],vertices[j-3],vertices[1]);
    }


    //DISSOLVE VERTICES
    dissolve_vertex(vertices[2]);
    dissolve_vertex(vertices[3]);


    //MAKE CELLS
    for(int j=2; j<=cells[0]; j++) make_cell(cells[j]);



    //******************************************************************************
    delete [] vertices;
    delete [] edges;
    delete [] cells;
    //******************************************************************************

    return 0;
}
//****************************************************************************
