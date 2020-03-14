import lsdtopytools as lsd
import numpy as np
import pyntail as pyt

raster = lsd.raster_loader.load_raster("putna_50_NDF.tif")
fdir = pyt.cppintail(raster["x_min"], raster["x_max"], raster["y_min"],raster["y_max"],
                     raster["res"], raster["res"],  raster["nrows"], raster["ncols"], -9999)
fdir.compute_neighbors(raster["array"])
fdirarray = fdir.get_flowdir()
fdir.find_nodes_with_no_donors(raster["array"])