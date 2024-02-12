import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VectorVisualizer:
    def __init__(self):
        # 3D 그래프 생성
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')        
        self.add_base_axes_to_figure()
        # 축 범위 설정
        self.ax.set_xlim(-500, 500)
        self.ax.set_ylim(-200, 300)
        self.ax.set_zlim(-100, 1000)
        # self.show_figure()
    
    def add_point_to_figure(self, pts):
        self.ax.scatter(*pts, c='c', marker='o', s=15, cmap='Greens')
    
    def add_base_axes_to_figure(self):
        origin = np.array([0,0,0])
        x = np.array([1,0,0])*20
        y = np.array([0,1,0])*20
        z = np.array([0,0,1])*20
        self.ax.quiver(*origin, *x, color='r', label='x')
        self.ax.quiver(*origin, *y, color='g', label='y')
        self.ax.quiver(*origin, *z, color='b', label='z')
        # 벡터 레이블 설정
        self.ax.text(*x, 'x', color='r')
        self.ax.text(*y, 'y', color='g')
        self.ax.text(*z, 'z', color='b')
        
    def add_vector_in_figure(self, origin:np.ndarray, v1:np.ndarray, v2:np.ndarray,\
        v3:np.ndarray, num:int):
        
        # 벡터를 그래프에 표시
        self.ax.quiver(*origin, *v1, color='r')
        self.ax.quiver(*origin, *v2, color='g')
        self.ax.quiver(*origin, *v3, color='b')
        # 벡터 레이블 설정
        self.ax.text(*(origin+v1), f'x{num}', color='r')
        self.ax.text(*(origin+v2), f'y{num}', color='g')
        self.ax.text(*(origin+v3), f'z{num}', color='b')
            
    def show_figure(self):
        # 축 레이블 설정
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 그래프 표시
        plt.legend()
        plt.show()
