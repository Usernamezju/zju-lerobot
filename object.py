# object.py
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

class SceneObject:
    def __init__(self, visualizer):
        self.vis = visualizer
        self.objects = {}

    def add_target(self, name, position, size=0.02, color=0x00ff00):
        """
        添加目标物体（默认绿色小方块）
        position: [x, y, z]
        size: 边长
        """
        self.vis[name].set_object(
            g.Box([size, size, size]),
            g.MeshLambertMaterial(color=color, opacity=1.0)
        )
        T = tf.translation_matrix(position)
        self.vis[name].set_transform(T)
        self.objects[name] = {
            'position': np.array(position),
            'size': size,
            'type': 'target'
        }

    def add_place_box(self, name, position, size=[0.08, 0.08, 0.04], color=0x0000ff):
        """
        添加放置盒子（默认蓝色）
        """
        self.vis[name].set_object(
            g.Box(size),
            g.MeshLambertMaterial(color=color, opacity=0.5)
        )
        T = tf.translation_matrix(position)
        self.vis[name].set_transform(T)
        self.objects[name] = {
            'position': np.array(position),
            'size': size,
            'type': 'place_box'
        }

    def get_position(self, name):
        """获取物体位置"""
        return self.objects[name]['position']

    def get_all(self):
        """返回所有物体"""
        return self.objects