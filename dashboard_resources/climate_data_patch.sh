
/p/projects/clme/menz/work/lecture/E-OBS_v25.0e_fillup/02.swim_units

# grids in m.swim/test grassdb
v.in.ascii in=eobs_centroids.csv out=eob_grid sep=comma x=3 y=2 skip=1 \
    col="station_id,lat double, lon double, height double" --o
v.proj in=eob_grid map=PERMANENT loc=lonlat
m.swim.climate gridfilepath=eobs_subbasins_grid.csv -d grid=eob_grid subbasins=subbasins \
    lon_column=lon lat_column=lat lonlat_precision=3 --o
#         lon     lat
# min  11.614  50.099
# max  12.157  50.428

# MDK centroids from Sachsen GRASS
v.proj in=MDK_centroids map=PERMANENT loc=lonlat db=./GRASS
m.swim.climate -d grid=MDK_centroids subbasins=subbasins gridfile=MDK_subbasin_grid.csv \
    lon_column=rlon lat_column=rlat lonlat_precision=3 --o
#        lon    lat
# min -4.065 -0.495
# max -3.735 -0.165