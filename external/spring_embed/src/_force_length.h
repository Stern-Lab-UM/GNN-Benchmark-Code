//****************************************************************************
//****************************************************************************
//****************************************************************************
//****************************************************************************
/*
 * edge_length: Compute or update edge-level topology/geometry information.
 * Parameters: int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
double edge_length(int i){
    double ddx,ddy;
    double *dxdydz = new double[3]; dxdydz[1]=0; dxdydz[2]=0;
    int v1=e[i][1], v2=e[i][2];
    torus_dx_dy_dz(dxdydz,v1,v2);
    ddx=v[v2][1]-(v[v1][1]+dxdydz[1]);
    ddy=v[v2][2]-(v[v1][2]+dxdydz[2]);
    delete []dxdydz;
    return sqrt(ddx*ddx+ddy*ddy);
}
//****************************************************************************
/*
 * e_length_force: Compute force contributions for the current vertex-model state.
 * Parameters: int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
double e_length_force(int i){

    double ddx,ddy;
    double *dxdydz = new double[3]; dxdydz[1]=0; dxdydz[2]=0;
    int v1=e[i][1], v2=e[i][2];
    torus_dx_dy_dz(dxdydz,v1,v2);
    ddx=v[v2][1]-(v[v1][1]+dxdydz[1]);
    ddy=v[v2][2]-(v[v1][2]+dxdydz[2]);
    delete []dxdydz;

    double c0=-e_g[i]/e_length[i];

    //v1
    v_F[v1][1]+=-c0*ddx;
    v_F[v1][2]+=-c0*ddy;

    //v2
    v_F[v2][1]+=c0*ddx;
    v_F[v2][2]+=c0*ddy;

    return e_g[i]*e_length[i];
}
//****************************************************************************
/*
 * e_spring_force: Compute force contributions for the current vertex-model state.
 * Parameters: int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
double e_spring_force(int i){

    double ddx,ddy;
    double *dxdydz = new double[3]; dxdydz[1]=0; dxdydz[2]=0;
    int v1=e[i][1], v2=e[i][2];
    torus_dx_dy_dz(dxdydz,v1,v2);
    ddx=v[v2][1]-(v[v1][1]+dxdydz[1]);
    ddy=v[v2][2]-(v[v1][2]+dxdydz[2]);
    delete []dxdydz;

    double c0=-2*kS*(e_length[i]-e_l0[i])/e_length[i];

    //v1
    v_F[v1][1]+=-c0*ddx;
    v_F[v1][2]+=-c0*ddy;

    //v2
    v_F[v2][1]+=c0*ddx;
    v_F[v2][2]+=c0*ddy;

    return kS*pow(e_length[i]-e_l0[i],2);
}
//****************************************************************************
/*
 * cell_perimeter: Compute or update cell-level topology/geometry information.
 * Parameters: int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
double cell_perimeter(int i){
    double _cperimeter=0.;
    for(int j = 1; j <= c[i][0]; ++j) _cperimeter += e_length[abs(c[i][j])];
    return _cperimeter;
}
//****************************************************************************
/*
 * e_perimeter_force: Compute force contributions for the current vertex-model state.
 * Parameters: int i, double cperimeter.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void e_perimeter_force(int i, double cperimeter){

    double ddx,ddy;

    double *dxdydz = new double[3]; dxdydz[1]=0; dxdydz[2]=0;
    int v1=e[i][1], v2=e[i][2];
    torus_dx_dy_dz(dxdydz,v1,v2);
    ddx=v[v2][1]-(v[v1][1]+dxdydz[1]);
    ddy=v[v2][2]-(v[v1][2]+dxdydz[2]);
    delete []dxdydz;

    //GRADIENT
    const double c0 = -2.*(cperimeter-P0)/e_length[i];

    v_F[v1][1] += -c0*ddx;
    v_F[v1][2] += -c0*ddy;

    v_F[v2][1] += c0*ddx;
    v_F[v2][2] += c0*ddy;

}
//****************************************************************************
/*
 * c_Perimeter_force: Compute force contributions for the current vertex-model state.
 * Parameters: int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
double c_Perimeter_force(int i){
    double _cperimeter=cell_perimeter(i);
    for(int j=1; j <= c[i][0]; j++) e_perimeter_force(abs(c[i][j]),_cperimeter);
    return pow(_cperimeter-P0,2);
}
//****************************************************************************
