"""These are test settings of SWIMpy."""

grass_db = "../grassdb"
grass_location = "utm32n"
grass_mapset =  "swim"
grass_setup = dict(elevation = "elevation@PERMANENT",
                   stations = "stations@PERMANENT",
                   upthresh=40, lothresh=11,
                   landuse = "landuse@PERMANENT",
                   soil = "soil@PERMANENT")
