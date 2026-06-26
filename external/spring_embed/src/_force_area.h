//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
double dA_new(int cellID, int j){

    int vert_ref_id=c_vertices[cellID][1];
    double vrefx = v[vert_ref_id][1];
    double vrefy = v[vert_ref_id][2];


    //vertices
    int v1, v2;
    if(j==c_vertices[cellID][0]){
        v1=c_vertices[cellID][j];
        v2=c_vertices[cellID][1];
    }
    else {
        v1=c_vertices[cellID][j];
        v2=c_vertices[cellID][j+1];
    }

    //v1
    double v1x=v[v1][1];
    double v1y=v[v1][2];
    //x
    if(fabs(v1x-vrefx)>perioXYZ[1]/2.){
        if(v1x<vrefx) v1x+=perioXYZ[1];
        else if(v1x>vrefx) v1x-=perioXYZ[1];
    }
    //y
    if(fabs(v1y-vrefy)>perioXYZ[2]/2.){
        if(v1y<vrefy) v1y+=perioXYZ[2];
        else if(v1y>vrefy) v1y-=perioXYZ[2];
    }

    //v2
    double v2x=v[v2][1];
    double v2y=v[v2][2];
    //x
    if(fabs(v2x-vrefx)>perioXYZ[1]/2.){
        if(v2x<vrefx) v2x+=perioXYZ[1];
        else if(v2x>vrefx) v2x-=perioXYZ[1];
    }
    //y
    if(fabs(v2y-vrefy)>perioXYZ[2]/2.){
        if(v2y<vrefy) v2y+=perioXYZ[2];
        else if(v2y>vrefy) v2y-=perioXYZ[2];
    }

    return 0.5*(-v2x*v1y + v1x*v2y);
}
//****************************************************************************
double CellArea_new(int i){
    double areaSum=0;
    for(int j=1; j<=c_vertices[i][0]; j++ ) areaSum += dA_new(i,j);
    return areaSum;
}
//****************************************************************************
double c_Adot_j(int i, int j){

    int vert_ref_id=c_vertices[i][1];
    double vrefx = v[vert_ref_id][1];
    double vrefy = v[vert_ref_id][2];

    //VERTEX IDs
    int v1, v2, v3;
    if(j==1){
        v1=c_vertices[i][c_vertices[i][0]];
        v2=c_vertices[i][1];
        v3=c_vertices[i][2];
    }
    else if(j==c_vertices[i][0]){
        v1=c_vertices[i][j-1];
        v2=c_vertices[i][j];
        v3=c_vertices[i][1];
    }
    else {
        v1=c_vertices[i][j-1];
        v2=c_vertices[i][j];
        v3=c_vertices[i][j+1];
    }

    //FORCES
    double forceX=v_F[v2][1];
    double forceY=v_F[v2][2];

    //VERTEX COORDINATES
    //v1
    double v1x=v[v1][1];
    double v1y=v[v1][2];
    //x
    if(fabs(v1x-vrefx)>perioXYZ[1]/2.){
        if(v1x<vrefx) v1x+=perioXYZ[1];
        else if(v1x>vrefx) v1x-=perioXYZ[1];
    }
    //y
    if(fabs(v1y-vrefy)>perioXYZ[2]/2.){
        if(v1y<vrefy) v1y+=perioXYZ[2];
        else if(v1y>vrefy) v1y-=perioXYZ[2];
    }

    //v3
    double v3x=v[v3][1];
    double v3y=v[v3][2];
    //x
    if(fabs(v3x-vrefx)>perioXYZ[1]/2.){
        if(v3x<vrefx) v3x+=perioXYZ[1];
        else if(v3x>vrefx) v3x-=perioXYZ[1];
    }
    //y
    if(fabs(v3y-vrefy)>perioXYZ[2]/2.){
        if(v3y<vrefy) v3y+=perioXYZ[2];
        else if(v3y>vrefy) v3y-=perioXYZ[2];
    }

    return forceX*(v3y-v1y)-forceY*(v3x-v1x);
}
//****************************************************************************
double c_AdotCalc(int i){
    double sum=0;
    for(int j=1; j<=c_vertices[i][0]; j++) sum+=c_Adot_j(i,j);
    return 0.5*sum;
}
//****************************************************************************
void v_AreaCompressibility_force_New(int j, int cellID, double _Cellarea){

    int vert_ref_id=c_vertices[cellID][1];
    double vrefx = v[vert_ref_id][1];
    double vrefy = v[vert_ref_id][2];


    //vertices
    int v1, v2;
    if(j==c_vertices[cellID][0]){
        v1=c_vertices[cellID][j];
        v2=c_vertices[cellID][1];
    }
    else {
        v1=c_vertices[cellID][j];
        v2=c_vertices[cellID][j+1];
    }

    //v1
    double v1x=v[v1][1];
    double v1y=v[v1][2];
    //x
    if(fabs(v1x-vrefx)>perioXYZ[1]/2.){
        if(v1x<vrefx) v1x+=perioXYZ[1];
        else if(v1x>vrefx) v1x-=perioXYZ[1];
    }
    //y
    if(fabs(v1y-vrefy)>perioXYZ[2]/2.){
        if(v1y<vrefy) v1y+=perioXYZ[2];
        else if(v1y>vrefy) v1y-=perioXYZ[2];
    }

    //v2
    double v2x=v[v2][1];
    double v2y=v[v2][2];
    //x
    if(fabs(v2x-vrefx)>perioXYZ[1]/2.){
        if(v2x<vrefx) v2x+=perioXYZ[1];
        else if(v2x>vrefx) v2x-=perioXYZ[1];
    }
    //y
    if(fabs(v2y-vrefy)>perioXYZ[2]/2.){
        if(v2y<vrefy) v2y+=perioXYZ[2];
        else if(v2y>vrefy) v2y-=perioXYZ[2];
    }


    double _deriv = -2.*kA*(_Cellarea-1.);


    //v1
    v_F[v1][1] += _deriv*(v2y/2.);
    v_F[v1][2] += _deriv*(-v2x/2.);
    //v2
    v_F[v2][1] += _deriv*(-v1y/2.);
    v_F[v2][2] += _deriv*(v1x/2.);

}
//****************************************************************************
double c_AreaCompressibility_force_New(int i){
    double _Cellarea=c_area[i];
    for(int j=1; j<=c_vertices[i][0]; j++ ) v_AreaCompressibility_force_New(j,i,_Cellarea);
    return kA*pow(_Cellarea-1.,2);
}
//****************************************************************************
