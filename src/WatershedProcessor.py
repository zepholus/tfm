import xarray as xr
from config.config import DATA_MODEL_DIR, OBSERVACIONS_FILTRAT_DIR, DATA_DIR
from pysheds.grid import Grid
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import pandas as pd
import rasterio as rio
from pyproj import Transformer
from functools import reduce
import copy


def plot_catchhment_delineation(catch_view, grid):
    # Plot the catchment
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)

    plt.grid('on', zorder=0)
    im = ax.imshow(np.where(catch_view, catch_view, np.nan), extent=grid.extent,
                    zorder=1, cmap='Greys_r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Delineated Catchment', size=14)

        
def accumulation_map(acc):
    # Compute flow accumulation based on computed flow direction
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    im = ax.imshow(acc, zorder=2,
                cmap='cubehelix',
                norm=colors.LogNorm(1, acc.max()),
                interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Upstream Cells')
    plt.title('Flow Accumulation', size=14)
    plt.tight_layout()

        
    
def fill_na(df):

    df[df < 0] = np.nan
    return df.fillna(df.groupby(df.index.month).transform('mean'))        


#get data of statis
def get_stations_data(stations_df, root_dir = DATA_MODEL_DIR / 'swat files 2022'):
    
    stations_timeseries = []

    #iterate over stations and get data
    for row in stations_df.itertuples():

        station = row.NAME

        station_path =  root_dir / f'{station}.txt'
        station_data = pd.read_csv(station_path)

        #add datetime index to dataframe, beginning at 1/1/2000
        station_data['datetime'] = pd.date_range(start='1/1/2000', periods=len(station_data), freq='D')
        station_data = station_data.rename(columns={'20000101': station}).set_index('datetime')

        station_data[station_data < 0] = np.nan

        stations_timeseries.append(station_data)
    
    
    #merge dataframes on index and return
    return reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), stations_timeseries)



#number of occurences of each cell type
def percentage_of_cells_in_catchment(raster, mask, lookup, column_name):
        
    masked = np.where(mask, raster, np.nan).reshape(-1)
    map_catchment = masked[~np.isnan(masked)]

    # Calculate the unique elements and their counts
    unique_elements, counts = np.unique(map_catchment, return_counts=True)


    df = pd.DataFrame({'Number': unique_elements, 'Count': counts})

    #convert column number to int and make index
    df['Number'] = df['Number'].astype(int)
    df = df.set_index('Number')


    #add column with percentage
    df['Percentage'] = df['Count'] / df['Count'].sum() * 100

    df[column_name] = df.index.map(lookup)

    types_to_add = []

    #if there are soil types that are in lookup but not in catchment, add them with count 0
    for cell_type in lookup.keys():
        if cell_type not in df[column_name].index:
            types_to_add.append({
                'Number': cell_type,
                'Count': 0,
                'Percentage': 0,
                column_name: lookup[cell_type]
            })
    
    if len(types_to_add) > 0:
        df_aux = pd.DataFrame(types_to_add).set_index('Number')
        df = pd.concat([df, df_aux])

    return df


class Watershed:
    def __init__(
            self, 
            grid, 
            fdir, 
            catch, 
            dem_8857, 
            soil_map_reprojected, 
            corinne_map_reprojected,
            slope,
            humidity_stations,
            precipitation_stations,
            wind_stations,
            solar_stations,
            temperature_stations,

            ):
        

        self.grid = grid
        self.fdir = fdir
        self.acc = self.grid.accumulation(self.fdir)
        self.catch = catch
        self.catch_view = grid.view(catch)
        self.dem_8857 = dem_8857
        self.corinne_map_reprojected = corinne_map_reprojected
        self.soil_map_reprojected = soil_map_reprojected
        self.slope = slope
        self.humidity_stations = humidity_stations
        self.precipitation_stations = precipitation_stations
        self.wind_stations = wind_stations
        self.solar_stations = solar_stations
        self.temperature_stations = temperature_stations


                                        


    def plot_catchhment_delineation(self):
        # Plot the catchment
        plot_catchhment_delineation(self.catch_view, self.grid)

    def accumulation_map(self):
        # Compute flow accumulation based on computed flow direction
        accumulation_map(self.acc)


    def percentage_soil_type(self):

        soil_type_lookup = {
            0: 'criic',
            1: 'udic',
            2: 'ustic',
            3: 'xeric',
            4: 'aridic',
            5: 'aquic',
        }

        return percentage_of_cells_in_catchment(self.soil_map_reprojected, self.catch, soil_type_lookup, 'Soil type')


    
    def percentages_land_use(self):

        land_use_lookup = {
            1: 'AGRH',
            2: 'AGRLL',
            3: 'FRST',
            4: 'RNGB',
            5: 'URHD',
            6: 'URLD',
            7: 'WATR',
        }

        return percentage_of_cells_in_catchment(self.corinne_map_reprojected, self.catch, land_use_lookup, 'Land use')

    
    def avg_height(self):

        #avg height of catchment
        masked = np.where(self.catch, self.dem_8857, np.nan).reshape(-1)
        height_catchment = masked[~np.isnan(masked)]

        height_catchment = np.average(height_catchment)
        return height_catchment
    
    def catchment_area(self):
        #number of cells of catchment
        mask = self.catch.reshape(-1)
        n_cells = np.count_nonzero(mask)

        resolution = self.dem_8857.rio.resolution()
        area = abs(n_cells * resolution[0] * resolution[1])
        return area / 1000000 #convert to km2
    
    def catchment_slope(self):                  
        masked = np.where(self.catch, self.slope, np.nan).reshape(-1)
        slope_catchment = masked[~np.isnan(masked)]

        slope_catchment = np.average(slope_catchment)
        return slope_catchment


    
    def _perform_avgs(self, df, name):
        #replace all values below 0 as nan
        df[df < 0] = np.nan

        #avg by rows
        df = df.mean(axis=1, numeric_only=True, skipna=True)
        
        #make datetime index
        df = df.to_frame()
        df = df.rename(columns={0: name})

        fill_avg_df = df.fillna(df.groupby(df.index.month).transform('mean'))        

        return fill_avg_df
    

    def get_humidity_stations_data(self):
        df =  get_stations_data(self.humidity_stations)
        return self._perform_avgs(df, 'humidity')
    
    def get_precipitation_stations_data(self):
        df = get_stations_data(self.precipitation_stations)
        return self._perform_avgs(df, 'precipitation')

    def get_wind_stations_data(self):
        df = get_stations_data(self.wind_stations)
        return self._perform_avgs(df, 'wind')

    def get_solar_stations_data(self):
        df = get_stations_data(self.solar_stations)
        return self._perform_avgs(df, 'solar')
    
    def get_temperature_stations_data(self):
        df = get_stations_data(self.temperature_stations)
        return self._perform_avgs(df, 'temperature')

    @staticmethod
    def get_station_streamflow(estacio_nom, swat_predictions):      
        df = pd.read_csv(OBSERVACIONS_FILTRAT_DIR / f"{estacio_nom}.csv")

        #rename Date column to datetime
        df = df.rename(columns={'Date': 'datetime'})

        #make datetime index
        df['datetime'] = pd.to_datetime(df['datetime'])

        #set datetime as index
        df = df.set_index('datetime')

        #fill with swat predictions
        if swat_predictions is not None:
            return df.fillna(swat_predictions)    
        else: 
            return df
    



    def get_statistics(self, estacio_nom, swat_predictions):
        #humid = self.get_humidity_stations_data()
        precip = self.get_precipitation_stations_data()
        #wind = self.get_wind_stations_data()
        #solar = self.get_solar_stations_data()
        streamflow = self.get_station_streamflow(estacio_nom, swat_predictions)
        temperature = self.get_temperature_stations_data()


        #merge by datetime
        df = pd.merge(temperature, precip, on='datetime')
        df = pd.merge(df, streamflow, on='datetime')
        #df = pd.merge(df, solar, on='datetime')

        #df = pd.merge(df, humid, on='datetime')
        #df = pd.merge(df, wind, on='datetime')

        df['area'] = self.catchment_area()
        df['avg_height'] = self.avg_height()
        df['avg_slope'] = self.catchment_slope()

        lands_uses = self.percentages_land_use()
        land_uses_dict = {}

        #for each index of land use, add a column with the percentage of that land use in the catchment
        for index, row in lands_uses.iterrows():
            land_uses_dict[f"{row['Land use']}_landuse"] = row['Percentage']

        soils = self.percentage_soil_type()
        land_soils_dict = {}

        #for each index of soil type, add a column with the percentage of that soil type in the catchment
        for index, row in soils.iterrows():
            land_soils_dict[f"{row['Soil type']}_soiltype"] = row['Percentage']

        df = df.assign(**land_uses_dict)
        df = df.assign(**land_soils_dict)


        return df



                


class WatershedProcessor:

    def __init__(
            self, 
            dem_path, 
            soil_map_path, 
            corinne_path, 
            coords_system =  "EPSG:8857",
            humidity_stations_path = DATA_MODEL_DIR / 'swat files 2022' / 'rh.txt',
            precipitation_stations_path = DATA_MODEL_DIR / 'swat files 2022' / 'pcp.txt',
            wind_stations_path = DATA_MODEL_DIR / 'swat files 2022' / 'wind.txt',
            solar_stations_path = DATA_MODEL_DIR / 'swat files 2022' / 'solar.txt',
            temperature_stations_path = DATA_MODEL_DIR / 'swat files 2022' / 'tmp.txt',
            ):
        
        #already converted to epsg:8857 for mantaining equal area between cells of raster
        dem_8857 = xr.open_dataarray(dem_path)
        soil_map = xr.open_dataarray(soil_map_path)
        corinne_map = xr.open_dataarray(corinne_path)


        self.dem_8857 = dem_8857
        self.soil_map_reprojected = soil_map.rio.reproject_match(dem_8857).assign_coords({
            "x": dem_8857.x,
            "y": dem_8857.y,
        })
        self.corinne_map_reprojected = corinne_map.rio.reproject_match(dem_8857).assign_coords({
            "x": dem_8857.x,
            "y": dem_8857.y,
        })

        grid = Grid.from_raster(str(dem_path))
        self.grid = grid    

        dem = grid.read_raster(str(dem_path))
        
        # Fill pits
        pit_filled_dem = grid.fill_pits(dem)

        # Fill depressions
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        
        # Detect flats
        inflated_dem = grid.resolve_flats(flooded_dem)

        # Specify directional mapping
        self.dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

        # Compute flow direction based on corrected DEM
        self.fdir = grid.flowdir(inflated_dem)

        #compute slope
        self.slope = grid.cell_slopes(inflated_dem, self.fdir)

        self.transformer_4326_to_8857 = Transformer.from_crs("EPSG:4326", coords_system)

        self.acc = self.grid.accumulation(self.fdir)
        self.humidity_stations = pd.read_csv(humidity_stations_path, index_col=0)
        self.precipitation_stations = pd.read_csv(precipitation_stations_path, index_col=0)
        self.wind_stations = pd.read_csv(wind_stations_path, index_col=0)
        self.solar_stations = pd.read_csv(solar_stations_path, index_col=0)
        self.temperature_stations = pd.read_csv(temperature_stations_path, index_col=0)
    
    def find_watershed(self, lat, lon):
        _grid_aux = copy.deepcopy(self.grid)

        # Specify pour point
        #x, y = 156681, 5165200

        x, y = self.transformer_4326_to_8857.transform(lat, lon)

        # Snap pour point to high accumulation cell
        x_snap, y_snap = _grid_aux.snap_to_mask(self.acc > 1000, (x, y))


        # Delineate the catchment
        catch = _grid_aux.catchment(x=x_snap, y=y_snap, fdir=self.fdir, xytype='coordinate')

        # Clip the bounding box to the catchment
        _grid_aux.clip_to(catch)

        return Watershed(
            _grid_aux, 
            self.fdir, 
            catch, 
            self.dem_8857, 
            self.soil_map_reprojected, 
            self.corinne_map_reprojected,
            self.slope,
            self._stations_in_catchment(self.humidity_stations, catch),
            self._stations_in_catchment(self.precipitation_stations, catch),
            self._stations_in_catchment(self.wind_stations, catch),
            self._stations_in_catchment(self.solar_stations, catch),
            self._stations_in_catchment(self.temperature_stations, catch),
            )
    
    def _is_coord_in_catchment(self, lat, long, catch):
        x8857, y8857 = self.transformer_4326_to_8857.transform(lat, long)
        rows, cols = rio.transform.rowcol(self.dem_8857.rio.transform(), x8857, y8857) #Get row and column of the given point
        return catch[rows, cols]    #Get value of the catchment (true/false) at given row and column
    
    def _stations_in_catchment(self, df, catch):
        _df = df.copy()
        _df['in_catchment'] = _df.apply(lambda row: self._is_coord_in_catchment(row['LAT'], row['LONG'], catch), axis=1)
        return _df[_df['in_catchment'] == 1].drop(columns=['in_catchment'])

        
    def accumulation_map(self):
        # Compute flow accumulation based on computed flow direction
        accumulation_map(self.acc)
    



    




