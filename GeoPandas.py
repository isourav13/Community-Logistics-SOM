import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

import contextily as cx



df = pd.read_csv('/Users/isourav/Documents SSD/Assignment/Dissertation/Dataset/Input_Data.csv')
df['Weight'] = np.random.randint(1, 5, df.shape[0])*df.iloc[:, 1]

#https://towardsdatascience.com/plotting-maps-with-geopandas-428c97295a73


nyc = gpd.read_file('/Users/isourav/Documents SSD/Assignment/Dissertation/Dataset/Shape/geo_export_900c7e2d-c78c-44d4-8d41-4ee68d2d5b9f.shp')
df_wm=nyc.to_crs(epsg=4326)
crs = {'init':'EPSG:4326'}
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
geo_df = gpd.GeoDataFrame(df, 
                          crs = crs, 
                          geometry = geometry)

geo_df.head()





fig, ax = plt.subplots(figsize = (10,10))
nyc.to_crs(epsg=4326).plot(ax=ax, color='WHITE')
geo_df.plot(ax=ax)
ax.set_title('New York')
cx.add_basemap(ax)


ax = df_wm.plot(figsize=(10, 10), alpha=0.3, edgecolor='BLACK', linewidth=2)
cx.add_basemap(ax)
geo_df.plot(ax=ax,color='BLUE')
ax.set_title('New York')




ax = df_wm.plot(figsize=(10, 10), alpha=0.3, edgecolor='BLACK', linewidth=2)
cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite)
cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels, crs=df_wm.crs.to_string())
geo_df.plot(ax=ax,color='BLUE')
ax.set_title('New York')




ax = df_wm.plot(figsize=(10, 10), alpha=0.3, edgecolor='BLACK', linewidth=2)
cx.add_basemap(ax, crs=df_wm.crs.to_string(), source=cx.providers.Stamen.Watercolor, zoom=12)
cx.add_basemap(ax, crs=df_wm.crs.to_string(), source=cx.providers.Stamen.TonerLabels, zoom=10)
geo_df.plot(ax=ax,color='BLUE')
ax.set_title('New York City')

