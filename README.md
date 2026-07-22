# SLAM y navegación autónoma con TurtleBot3

Sistema autónomo de localización, mapeo y navegación desarrollado en **Python y ROS 2** para un robot **TurtleBot3** simulado en **Gazebo**.

El proyecto integra odometría, mediciones LIDAR, FastSLAM, planificación de rutas mediante D* Lite y control de movimiento para explorar un entorno, construir un mapa y navegar hacia diferentes objetivos evitando obstáculos.

Fue desarrollado como proyecto final de la materia **Principios de la Robótica Autónoma** de la carrera de Ingeniería en Inteligencia Artificial de la Universidad de San Andrés.

## Objetivos

- Estimar la pose del robot utilizando odometría y sensores con ruido.
- Construir un mapa del entorno mediante FastSLAM.
- Localizar el robot dentro de un mapa previamente generado.
- Planificar rutas seguras sobre una grilla de ocupación.
- Seguir la trayectoria calculada mediante un controlador de movimiento.
- Replanificar cuando cambia el objetivo o aparecen nuevos obstáculos.
- Visualizar el mapa, la localización y las trayectorias mediante RViz.

## Tecnologías utilizadas

- Python 3
- ROS 2 Humble
- Gazebo
- TurtleBot3
- NumPy
- OpenCV
- Numba
- RViz 2
- FastSLAM
- Filtros de partículas
- D* Lite
- Sensores LIDAR y odometría

## Arquitectura

El sistema está organizado como un conjunto de nodos y componentes independientes de ROS 2.

| Módulo | Responsabilidad |
| --- | --- |
| `odometry.py` | Procesa la odometría y publica los incrementos de movimiento del robot. |
| `slam.py` | Implementa FastSLAM, procesa las mediciones LIDAR y estima simultáneamente la pose y el mapa. |
| `save_map.py` | Convierte la información obtenida durante SLAM en una grilla de ocupación. |
| `localization.py` | Localiza el robot dentro de un mapa conocido mediante un filtro de partículas. |
| `d_star_lite.py` | Calcula y actualiza rutas sobre la grilla de ocupación mediante D* Lite. |
| `path_planning.py` | Integra la posición estimada, el objetivo, el mapa y el planificador de caminos. |
| `navigation.py` | Ejecuta el seguimiento de trayectoria, publica comandos de velocidad y responde ante obstáculos. |

Flujo general del sistema:

```text
Odometría + LIDAR
        |
        v
 FastSLAM y mapeo
        |
        v
   Localización
        |
        v
Planificación D* Lite
        |
        v
Control y navegación
```

## Funcionalidades principales

### SLAM

- Representación probabilística de la pose mediante partículas.
- Actualización de landmarks a partir de mediciones LIDAR.
- Manejo de incertidumbre en el movimiento y las observaciones.
- Extracción de segmentos y características del entorno.
- Construcción progresiva del mapa.
- Publicación de la pose estimada y elementos de depuración en RViz.

### Localización

- Estimación de la pose sobre un mapa previamente generado.
- Inicialización de la posición mediante la herramienta **2D Pose Estimate** de RViz.
- Actualización de partículas a partir de odometría y LIDAR.
- Publicación de la pose estimada en `/amcl_pose`.

### Planificación y navegación

- Planificación en ocho direcciones sobre una grilla de ocupación.
- Utilización de distancia octil como heurística.
- Replanificación incremental mediante D* Lite.
- Incorporación de obstáculos detectados durante la navegación.
- Seguimiento de la trayectoria calculada.
- Publicación de comandos de velocidad en `/cmd_vel`.
- Soporte para definir nuevas metas durante la ejecución.

## Requisitos

Entorno recomendado:

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo
- Paquetes de TurtleBot3
- `colcon`
- Python 3
- NumPy
- OpenCV
- Numba
- Paquetes `custom_msgs` y `turtlebot3_custom_simulation`

Instalación de dependencias generales:

```bash
sudo apt update
sudo apt install ros-humble-turtlebot3* \
    ros-humble-rviz2 \
    python3-numpy \
    python3-opencv \
    python3-pip

python3 -m pip install numba
```

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/Ciror3/tpFinalRobotica.git
cd tpFinalRobotica/tpFinalA/ros2_ws
```

Instalar las dependencias declaradas:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

Compilar el workspace:

```bash
colcon build --packages-select tpFinalA
source install/setup.bash
```

El workspace debe cargarse en cada terminal utilizada:

```bash
source install/setup.bash
```

## Ejecución

### 1. Iniciar la simulación

En una primera terminal:

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_custom_simulation custom_casa.launch.py
```

Para probar el sistema en un escenario con obstáculos:

```bash
ros2 launch turtlebot3_custom_simulation custom_casa_obs.launch.py
```

### 2. Ejecutar FastSLAM

En una nueva terminal:

```bash
cd tpFinalRobotica/tpFinalA/ros2_ws
source install/setup.bash
ros2 launch tpFinalA launch_fastslam.launch.py
```

Durante esta etapa, el sistema utiliza las mediciones publicadas en `/scan` y los incrementos de odometría publicados en `/delta`.

Los resultados pueden visualizarse en RViz mediante:

- `/fpose`
- `/fastslam/markers`
- `/map`
- `/scan`
- `/odom`

### 3. Explorar el entorno

El robot puede controlarse manualmente para recorrer el entorno y construir el mapa:

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```

Durante la exploración, FastSLAM estima la trayectoria del robot y actualiza los landmarks detectados mediante el sensor LIDAR.

### 4. Guardar el mapa

Una vez explorado el entorno:

```bash
ros2 run tpFinalA save_map
```

El mapa generado será utilizado posteriormente para la localización y la navegación autónoma.

### 5. Ejecutar la navegación autónoma

Con el mapa generado disponible:

```bash
ros2 launch tpFinalA launch_automatic_navigation.launch.py
```

En RViz:

1. Seleccionar **2D Pose Estimate** para indicar una estimación inicial de la posición.
2. Seleccionar **2D Goal Pose** para establecer el objetivo.
3. Observar la pose estimada y el camino calculado.
4. Definir una nueva meta para comprobar la capacidad de replanificación.
5. Incorporar obstáculos durante el recorrido para evaluar la actualización de la ruta.

## Tópicos relevantes

| Tópico | Descripción |
| --- | --- |
| `/scan` | Mediciones del sensor LIDAR. |
| `/delta` | Incrementos de movimiento obtenidos desde la odometría. |
| `/fpose` | Pose estimada durante FastSLAM. |
| `/amcl_pose` | Pose estimada durante la localización sobre el mapa. |
| `/map` | Grilla de ocupación del entorno. |
| `/planned_path` | Trayectoria calculada por el planificador. |
| `/global_path` | Trayectoria utilizada por el controlador. |
| `/cmd_vel` | Comandos de velocidad enviados al robot. |
| `/fastslam/markers` | Partículas y landmarks visualizados en RViz. |
| `/localization/markers` | Información visual del proceso de localización. |
| `/inflated_obstacles` | Representación de obstáculos con margen de seguridad. |

## Estructura del repositorio

```text
tpFinalRobotica/
├── PRA_TPFinal_Parte_A.pdf
├── PRA_TPFinal_Parte_B.pdf
└── tpFinalA/
    └── ros2_ws/
        └── src/
            └── tpFinalA/
                ├── launch/
                │   ├── launch_fastslam.launch.py
                │   └── launch_automatic_navigation.launch.py
                ├── test/
                ├── package.xml
                ├── setup.py
                └── tpFinalA/
                    ├── odometry.py
                    ├── slam.py
                    ├── save_map.py
                    ├── localization.py
                    ├── d_star_lite.py
                    ├── path_planning.py
                    └── navigation.py
```

## Algoritmos implementados

### FastSLAM

FastSLAM utiliza un conjunto de partículas para representar posibles estados del robot. Cada partícula mantiene:

- Posición y orientación estimadas.
- Peso asociado a la probabilidad de la hipótesis.
- Landmarks detectados.
- Media y covarianza de cada landmark.

La actualización se realiza combinando el modelo de movimiento obtenido mediante odometría con las observaciones proporcionadas por el sensor LIDAR.

### D* Lite

D* Lite permite calcular caminos sobre una grilla de ocupación y actualizar eficientemente la solución cuando cambia el entorno.

El planificador:

- Mantiene los valores `g` y `rhs` de cada celda.
- Utiliza una cola de prioridad.
- Considera movimientos horizontales, verticales y diagonales.
- Detecta obstáculos incorporados durante la navegación.
- Repara la ruta sin calcular nuevamente todo el camino desde cero.

## Resultados

El proyecto integra el ciclo completo de un sistema de navegación autónoma:

1. Percepción del entorno.
2. Estimación de movimiento.
3. Localización probabilística.
4. Construcción del mapa.
5. Planificación de rutas.
6. Seguimiento de trayectorias.
7. Detección de obstáculos.
8. Replanificación.

El sistema permite construir un mapa de un entorno inicialmente desconocido y utilizarlo posteriormente para navegar hacia metas arbitrarias.

## Demostración

Podés agregar una imagen o GIF del proyecto guardándolo dentro de una carpeta `docs`:

```markdown
![Demostración del sistema](docs/demo.gif)
```

También podés agregar un enlace a un video:

```markdown
[Ver demostración en video](https://www.youtube.com/URL_DEL_VIDEO)
```

## Mejoras futuras

- Evaluar cuantitativamente el error de localización.
- Medir la calidad del mapa generado.
- Comparar D* Lite con A*, Dijkstra y otros planificadores.
- Incorporar pruebas automatizadas para los componentes algorítmicos.
- Parametrizar el ruido y los valores del controlador mediante archivos YAML.
- Mejorar la documentación de los nodos y tópicos.
- Validar el sistema sobre un TurtleBot3 físico.
- Incorporar integración continua para compilar y probar el paquete ROS 2.

## Autor

**Ciro Russi**

Estudiante de Ingeniería en Inteligencia Artificial  
Universidad de San Andrés

[GitHub](https://github.com/Ciror3)

## Licencia

El paquete declara la licencia Apache 2.0. Consultar el archivo `package.xml` para más información.
