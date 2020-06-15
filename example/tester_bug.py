import CHONK_cpp as ch
import numpy as np
import fastscapelib_fortran as fs
import lsdtopytools as lsd
import numba as nb
from matplotlib import pyplot as plt

dt = 10
model = ch.ModelRunner( dt, 0, ["drainage_area","move","basic_SPIL"], "MF_fastscapelib") #"D8"
# model.update_int_param("n_rows", 5)




nrows = 100
ncols = 100
xres = 100
yres = 100
xmin = 0
xmax = ncols*yres
ymin = 0
ymax = nrows*xres

Z = np.load("test_animation/test_animation_0090.npy")
# for j in range(ncols):
#     arr = np.arange(0,round(nrows/2))*1
#     Z[0:round(ncols/2),j] +=arr
#     Z[round(ncols/2):,j] += arr[::-1] 


uplift = np.zeros_like(Z)
uplift[1:-1] = 0.001
# uplift[60:120,:] = 0.001*dt
uplift = uplift.ravel()

fs.fastscape_init()
fs.fastscape_set_nx_ny(ncols,nrows)
fs.fastscape_setup()
fs.fastscape_set_xl_yl(xmax - xmin, ymax - ymin)
fstctx = fs.fastscapecontext
fstctx.p = 3
fstctx.h = Z.astype(np.float64).ravel()
# fs.fastscape_init_h(raster["array"].astype(np.float64).ravel())

# fs.fastscape_init_h(raster["array"].astype(np.float64).ravel())
#     print("p")
fstctx.kf = np.ones_like(fstctx.h) * 1e-3
fstctx.kfsed = 1e-3
fstctx.m = 0.45
fstctx.n = 1.
fstctx.kd = np.ones_like(fstctx.h) * 1e-3
fstctx.kdsed = 1e-3
fstctx.g1 = 1.0
fstctx.g2 = 1.0
fstctx.ibc = 1010
fs.fastscape_set_dt(1000)

# model.update_array2d_double_param[] = 



# model.update_array_int_param("pre_stack", prestack)
# model.update_array_int_param("pre_rec", prerec)
# model.update_array_int_param("m_stack", mstack)
# model.update_array2d_int_param("m_rec", mrec)
# model.update_array2d_int_param("m_don", mdon)
model.update_array_double_param("surface_elevation", Z.ravel())
# model.update_array2d_double_param("length", lengths)
model.update_double_param("x_min", xmin)
model.update_double_param("y_min", ymin)
model.update_double_param("x_max", xmax)
model.update_double_param("y_max", ymax)
model.update_double_param("x_res", xres)
model.update_double_param("y_res", yres)
model.update_double_param("no_data", -9999)
model.update_int_param("n_rows", nrows)
model.update_int_param("n_cols", ncols)
model.update_int_param("n_elements", nrows*ncols)
model.update_array_double_param("surface_elevation_tp1", np.copy(Z.ravel()))
model.update_array_double_param("sed_height", np.zeros_like(Z.ravel()))
model.update_array_double_param("sed_height_tp1", np.zeros_like(Z.ravel()))
model.update_array_double_param("lake_depth", np.zeros_like(Z.ravel()))

model.update_double_param("SPIL_n", 1)
model.update_double_param("SPIL_m", 0.45)
model.update_array_double_param("erodibility_K", np.zeros_like(Z.ravel())+1e-4)



# print("y")
dt = 10
uplift = uplift.reshape(nrows,ncols)
uplift = np.zeros_like(uplift)
uplift[1:-1,:] = 0.002
uplift[10:20,:] = 0.005
# uplift[60:80,60:80] = 0
# uplift[50:60,50:60] = -0.2
uplift = uplift.ravel()

this_K = np.zeros_like(Z.ravel()).reshape(nrows,ncols)+1e-4
this_K[10:20,:] = 1e-5 
this_K = this_K.ravel()
# print("t")
model.update_timestep(dt)
for i in range(1):
    print("f")
#     print("o", end = "") #if i%10 ==0 else 0
    new_elev = model.get_array_double_param("surface_elevation_tp1")
    new_sed = model.get_array_double_param("sed_height_tp1")
    new_elev = new_elev + (uplift * dt)

    print("l")
#     fs.fastscape_init()
#     fs.fastscape_set_nx_ny(ncols,nrows)
#     fs.fastscape_setup()
#     fs.fastscape_set_xl_yl(xmax - xmin, ymax - ymin)
#     fstctx = fs.fastscapecontext
    fstctx.p = 3
    fstctx.h = new_elev
    # fs.fastscape_init_h(raster["array"].astype(np.float64).ravel())
    print("p")
    fstctx.kf = np.ones_like(fstctx.h) * 1e-3
    fstctx.kfsed = 1e-3
    fstctx.m = 0.45
    fstctx.n = 1.
    fstctx.kd = np.ones_like(fstctx.h) * 1e-3
    fstctx.kdsed = 1e-3
    fstctx.g1 = 1.0
    fstctx.g2 = 1.0
    fstctx.ibc = 1010
    fs.fastscape_set_dt(1000)
    print("1")
    fs.flowrouting_first_stack ()
    print("2")
    prestack = np.copy(fstctx.stack.astype('int') - 1)
    prerec = np.copy(fstctx.rec.astype('int') - 1)
    fs.flowrouting_reroute_local_minima ()
    print("3")
    postsrec = np.copy(fstctx.rec.astype('int') - 1)
    poststack = np.copy(fstctx.stack.astype('int') - 1)
    CHECK_POSTREC = np.copy(postsrec)
    postsrec, pit_to_process = ch.preprocess_stack(prestack, prerec, poststack,  postsrec)
    fstctx.rec = postsrec + 1
    fstctx.stack = poststack + 1
    print("4")
    fs.flowrouting_multiple_flow ()
    print("5")
    mdon = np.copy(fstctx.don.astype('int').transpose() - 1)
    mstack = np.copy(fstctx.mstack.astype('int') - 1)
    mrec = np.copy(fstctx.mrec.astype('int').transpose() - 1)
    lengths = np.copy(fstctx.mlrec.transpose())
    weights = np.copy(fstctx.mwrec.transpose())

#     fs.fastscape_destroy()
#     print("u")

    model.update_array_int_param("pre_stack", prestack)
    model.update_array_int_param("pre_rec", prerec)
    model.update_array_int_param("post_rec", postsrec)
    model.update_array_int_param("m_stack", mstack)
    model.update_array2d_int_param("m_rec", mrec)
    model.update_array2d_int_param("m_don", mdon)
    model.update_array2d_double_param("length", lengths)
    model.update_array2d_double_param("external_weigths_water", weights)
    model.update_array_int_param("depression_to_reroute", pit_to_process)


    model.update_array_double_param("surface_elevation", np.copy(new_elev))
    model.update_array_double_param("surface_elevation_tp1", np.copy(new_elev))
#     model.update_array_double_param("sed_height", np.zeros_like(new_sed.ravel()))
    model.update_array_double_param("sed_height", np.copy(new_sed))
    model.update_array_double_param("sed_height_tp1", np.copy(new_sed))
#     model.update_array_double_param("lake_depth", model.get_other_attribute("lake_depth"))

    model.update_array_double_param("erodibility_K", this_K)
    model.update_double_param("SPIL_n", 1)
    model.update_double_param("SPIL_m", 0.45)

# 
    
    print("r")
    model.DEBUG_check_weird_val_stacks()    
    print("bulf")
    model.initiate_nodegraph()
    print("b")
    model.run()
    if(i%10==0):
        name = str(i)
        while(len(name)<4):
           name = "0" + name
        np.save("test_animation/test_animation_%s.npy"%(name), new_elev)

    