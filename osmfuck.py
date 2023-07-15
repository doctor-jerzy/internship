import osmnx as ox
import math
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
import pyproj
import numpy as np
import os

# функция возвращает utm crs по географическим координатам (широта, долгота)
def convert_wgs_to_utm(lat: float, lon: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    prop_crs = pyproj.CRS.from_epsg(epsg_code)
    return prop_crs

class City:
    # вводится название города, сразу определяется пояс UTM для него
    def __init__(self, name): 
        self.name = name 
        self.city_gdf = ox.geocode_to_gdf(name)
        self.scr_crs = self.city_gdf.crs
        self.prop_crs = convert_wgs_to_utm(self.city_gdf['lat'].item(), 
                                           self.city_gdf['lon'].item())
        self.name_en = self.city_gdf['display_name'][0].split(',')[0]
        self.type = self.city_gdf['type'][0]
        self.buildings = None
    
    # функция получения геометрии требуемых типов зданий
    #(по умолчанию выводятся все здания)
    def get_buildings(self, types=True):  
        # получаем всю инфу о зданиях
        buildings = ox.features_from_place(self.name, {'building': types})
        buildings.sort_index(inplace=True)
        
        # удаляю лишние столбцы и строки, переименовываю те что оставляю,
        # так как столбцов в выгружаемом датасете ОЧЕНЬ МНОГО
        # оставил геометрию, тип здания и этажность
        buildings = buildings[['geometry', 'building', 'building:levels']]
        if 'node' in set(buildings.index.get_level_values(0).to_list()):
            buildings = buildings.drop('node')
        buildings.rename(columns={'building': 'type',
                                  'building:levels': 'levels'},
                         inplace=True)

        # делаю параметр этажности числовым
        buildings.levels = pd.to_numeric(buildings.levels, errors='coerce')

        # в осм не дана этажность всех зданий, поэтому как костыль берётся 
        # средняя этажность по каждому типу зданий, которая ставится вместо NaN
        # если для типа зданий этажности нет, удаляю
        for i in buildings['type'].unique():
            try:
                mean_level = buildings[buildings['type'] == i]['levels'].mean()
                mean_level = int(mean_level)
            except ValueError:
                buildings.drop(buildings[buildings['type'] == i].index,
                               inplace = True)
            
            buildings.loc[buildings['type'] == i, 'levels'] = buildings.loc[
                buildings['type'] == i, 'levels'].fillna(mean_level)
        
        # записываю в класс инфу о зданиях, чтоб ей потом пользоваться
        self.buildings = buildings

        # вывожу, если удобней непосредственно присвоить внешней переменной
        return self.buildings
        
    # функция подсчёта количества проживающих в зданиях людей
    # принимает словарь типов зданий, для которых считается количество людей
    # подсчёт ведётся по формуле относительно общей площади здания
    # по умолчанию считает людей во всех домах
    # если дан список, только в типах домов из списка
    # подразумевается, что считаются только жилые дома, хотя ограничений нет
    # если дома к этому шагу не сгенерированы, генерирует по заданному списку
    # если списка нет, генерирует все здания, считает все здания
    # присоединяет общую площадь и людей к self.buildings
    # выводит датасет с домами с площадью и людьми
    def count_people(self, types=True):
        # если сет домов ещё не был сгенерирован - генерируем
        if self.buildings is None:
            self.get_buildings(types)
        
        # обнуляю столбцы с площадью и людьми
        self.buildings['area'] = np.nan
        self.buildings['people'] = np.nan
        
        # выбираю нужные типы домов (жилые)
        if type(types) == list:
            buildings = self.buildings[self.buildings['type'].isin(types)]
        else:
            buildings = self.buildings
        
        # перевожу геометрию в нужную црс
        buildings = buildings.to_crs(self.prop_crs)
        
        # считаем население каждого жилого дома, по формуле
        # площадь дома * пониж. коэффициент / м2 на человека
        # в данном случае пониж. коэффициент = 0,4, 20м2 на человека
        buildings['area'] = buildings.area * buildings['levels'] * 0.4
        buildings['people'] = buildings['area'] / 20
        buildings.people = buildings.people.round()

        # если в доме 0 чел. (слишком маленькая площадь), меняем на 1
        buildings.loc[buildings.people == 0, 'people'] = 1
         
        # добавляю значения в датасет класса
        self.buildings['area'] = buildings['area']
        self.buildings['people'] = buildings['people']
        
        # вывожу датасет с посчитанными домами по поясу utm
        return buildings
        
    # строит сетку. размер по умолчанию ~537 метров
    # это расстояние от центра гексагона до его стороны (радиус впис. окр-ти)
    # получается, что диаметр в районе километра
    # можно задать свой радиус при желании
    def get_hex (self, size=np.sqrt(500000/np.sqrt(3)), *, top='flat'):
        """size - это радиус вписаной в гексагон окружности, то есть
        расстояние от центра до стороны гексагона
        по умолчанию size стоит такое, что одна ячейка по площади равна 1 км2"""
        
        if top not in ('flat', 'point'):
            raise ValueError("The 'top' parameter can only take the values " \
                             "'flat' or 'point'. The default setting is 'flat'.")
        
        # переводим в метровую crs
        city_gdf = self.city_gdf.to_crs(self.prop_crs)
        
        # определяем границы сетки как экстент по городу
        xmin, ymin, xmax, ymax = city_gdf.bounds.iloc[0]
        
        # создаём точки центров гексагонов
        # для этого сначала нужно создать прямоугольную сетку точек
        if top == 'flat':
            x, y = np.meshgrid(np.arange(xmin, xmax, size*np.sqrt(3)),
                               np.arange(ymin, ymax, size))
        elif top == 'point':   
            x, y = np.meshgrid(np.arange(xmin, xmax, size),
                               np.arange(ymin, ymax, size*np.sqrt(3)))
        points = np.dstack((x, y))
        
        # чтобы сетка стала гексагональной, удаляем точки,
        # которые не "попадают" в центры гексагонов 
        condition = np.sum(np.indices(points.shape[:2]), axis=0) % 2 == 0
        points = points[condition].tolist()
        
        # создаём вокруг точек точки вершин шестиугольников
        if top == 'flat':
            angles = [60 * i for i in range(6)]
        elif top == 'point':
            angles = [60 * i + 30 for i in range(6)]
        
        side_length = 2 * size / math.sqrt(3)
        hexagones = [
            [(point[0] + side_length * math.cos(math.radians(angle)),
              point[1] + side_length * math.sin(math.radians(angle)))
             for angle in angles] for point in points
            ]
        
        # создаём полигоны шестиугольников
        hexagones = [Polygon(hexagon) for hexagon in hexagones]
        
        # создаём геодатасекс
        gdf_hex = gpd.GeoDataFrame(geometry=hexagones, crs=self.prop_crs)
        
        # заношу сетку в класс, переводя в изначальную crs
        self.hex = gdf_hex.to_crs(self.scr_crs)
        
        # вывожу сетку в crs по поясу utm
        return gdf_hex
    
    
    # подсчёт числа людей в каждой ячейке
    def density_by_hex(self):     
        
        # перевод в utm, так как геометрия в градусах плохо работает
        buildings = self.buildings.to_crs(self.prop_crs)
        gdf_hex = self.hex.to_crs(self.prop_crs)
        
        #пересечение сетки и домов:
        buildings.geometry=buildings.centroid
        gdf_hex = gpd.sjoin(gdf_hex, buildings, how='left')

        # аггрегирование данных сетки, получение итогового геодатафрейма GRID
        gdf_hex=gdf_hex[['geometry', 'people']]
        gdf_hex.replace(np.nan, 0, inplace=True)
        gdf_hex.reset_index(inplace=True)
        gdf_hex = gdf_hex.groupby(['index']).agg({'geometry':'first',
                                                  'people':'sum'})
        gdf_hex = gdf_hex.set_geometry('geometry')
        gdf_hex.crs = self.prop_crs
        
        # запись в класс, уже в wgs 84
        self.hex = gdf_hex.to_crs(self.scr_crs)
        
        # возврат в utm, на случай если пользователю понадобится
        return gdf_hex
    
    # экспорт датасета ячеек
    # указывается формат, пока только 'csv' и 'geojson'
    def export_hex(self, formate='csv'):
        
        # путь экспорта
        dir_path = f'{self.type}/{self.name_en}'
        os.makedirs(dir_path, exist_ok=True)
        
        # непосредственно экспорт
        if formate == 'csv':
            self.hex.to_csv(f'{dir_path}/hex.csv')
        elif formate == 'geojson':
            self.hex.to_file(f'{dir_path}/hex.geojson', 'GeoJSON')
   


