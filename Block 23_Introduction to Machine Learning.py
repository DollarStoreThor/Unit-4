import pandas as pd

# Load the data
file_path = "real_estate_coordinates.csv"
df_properties = pd.read_csv(file_path)
df_properties.head()

import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def minkowski_distance(point1, point2, p=2):
    return np.power(np.sum(np.abs(point1 - point2)**p), 1/p)


p1 = np.array([36.87, -99.84])
p2 = np.array([36.87, -99.84])
x = euclidean_distance(p1, p2)

print(x)

p0 = np.array([df_properties['Latitude'][0], df_properties['Longitude'][0]])

df_properties['Euclidean_to_ID1'] = df_properties[['Latitude', 'Longitude']].apply(lambda pN: euclidean_distance(p0, pN), axis=1)
df_properties['Manhattan_to_ID1'] = df_properties[['Latitude', 'Longitude']].apply(lambda pN: manhattan_distance(p0, pN), axis=1)
df_properties['Minkowski_to_ID1'] = df_properties[['Latitude', 'Longitude']].apply(lambda pN: minkowski_distance(p0, pN, p=3), axis=1)
               
df_properties


print('\n~~~ EUCLIDEAN ~~~~ \n',df_properties.sort_values(by='Euclidean_to_ID1').head(6).tail(), '\n-----------------------------------------------------------------\n')
print('\n~~~ MANHATTAN ~~~ \n',df_properties.sort_values(by='Manhattan_to_ID1').head(6).tail(), '\n-----------------------------------------------------------------\n')
print('\n~~~ MINKOWSKI ~~~ \n',df_properties.sort_values(by='Minkowski_to_ID1').head(6).tail(), '\n-----------------------------------------------------------------\n')



