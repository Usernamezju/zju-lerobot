# obstacle.py
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

class ObstacleManager:
    def __init__(self, visualizer):
        self.vis = visualizer
        self.obstacles = {}  # 存储所有障碍物 {name: {position, size}}

    def add_box(self, name, position, size):
        """
        添加长方体障碍物
        position: [x, y, z] 中心位置
        size: [lx, ly, lz] 长宽高
        """
        self.vis[name].set_object(
            g.Box(size),
            g.MeshLambertMaterial(color=0xff0000, opacity=0.5)
        )
        T = tf.translation_matrix(position)
        self.vis[name].set_transform(T)
        self.obstacles[name] = {'position': position, 'size': size, 'type': 'box'}

    def add_sphere(self, name, position, radius):
        """
        添加球体障碍物
        position: [x, y, z] 中心位置
        radius: 半径
        """
        self.vis[name].set_object(
            g.Sphere(radius),
            g.MeshLambertMaterial(color=0xff0000, opacity=0.5)
        )
        T = tf.translation_matrix(position)
        self.vis[name].set_transform(T)
        self.obstacles[name] = {'position': position, 'radius': radius, 'type': 'sphere'}

    def remove(self, name):
        """删除障碍物"""
        self.vis[name].delete()
        del self.obstacles[name]

    def get_all(self):
        """返回所有障碍物"""
        return self.obstacles