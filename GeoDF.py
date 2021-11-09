import geopandas
import contextily as cx

import numpy as np

import pandas as pd
import geopandas
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/isourav/Documents SSD/Assignment/Dissertation/Dataset/Input_Data.csv')
df['Weight'] = np.random.randint(1, 5, df.shape[0])*df.iloc[:, 1]

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))


df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')



plt.show()
df.crs
df_wm = df.to_crs(epsg=3857)

ax = df_wm.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
cx.add_basemap(ax)

ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
cx.add_basemap(ax, crs=df.crs)

ax.set_title('New York')
ax = df_wm.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite)
ax.set_axis_off()