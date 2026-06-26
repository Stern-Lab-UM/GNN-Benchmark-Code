//****************************************************************************
//****************************************************************************
//*******************************TORUS****************************************
//****************************************************************************
//****************************************************************************
void torus_vertex(int i){
    //x
    if      (v[i][1]<0)             v[i][1] += perioXYZ[1];
    else if (v[i][1]>perioXYZ[1])   v[i][1] -= perioXYZ[1];
    //y
    if      (v[i][2]<0)             v[i][2] += perioXYZ[2];
    else if (v[i][2]>perioXYZ[2])   v[i][2] -= perioXYZ[2];
}
//****************************************************************************
void torus_dx_dy_dz(double *dxdydz, int vert_id, int vert_ref_id){

    dxdydz[1]=0; dxdydz[2]=0;

    //x
    if(fabs(v[vert_id][1]-v[vert_ref_id][1])>0.5*perioXYZ[1]){
        if      (v[vert_id][1]<v[vert_ref_id][1])   dxdydz[1]=perioXYZ[1];
        else if (v[vert_id][1]>v[vert_ref_id][1])   dxdydz[1]=-perioXYZ[1];
    }
    else dxdydz[1]=0;
    //y
    if(fabs(v[vert_id][2]-v[vert_ref_id][2])>0.5*perioXYZ[2]){
        if      (v[vert_id][2]<v[vert_ref_id][2])   dxdydz[2]=perioXYZ[2];
        else if (v[vert_id][2]>v[vert_ref_id][2])   dxdydz[2]=-perioXYZ[2];
    }
    else dxdydz[2]=0;
}
//****************************************************************************
