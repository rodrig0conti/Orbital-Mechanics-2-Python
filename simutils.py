import datetime as dt
import numpy as np
import os
from vispy.scene import MatrixTransform as Mat4
from vispy.util.quaternion import Quaternion as Quat

class Error(Exception):
    pass

class InvalidConstruction(Error):
    def __init__(self,message):
        self.message=message

class Quaternion:
    def __init__(self, arg1 = None, arg2 = None):
        if arg1 is None and arg2 is None:
            self.q = np.array([1,0,0,0])
        elif arg1 is not None and arg2 is None:
            if type(arg1) is Quaternion:
                self.q = np.array(arg1.q)
            elif len(arg1) == 4:
                self.q = np.array(arg1)
            elif len(arg1) == 3:
                self.q = np.array([0,*arg1])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        elif arg1 is not None and arg2 is not None:
            if len(arg2) == 3:
                mag = np.sqrt(arg2[0]**2.0+arg2[1]**2.0+arg2[2]**2.0)
                self.q = np.array([np.cos(arg1/2.0),*(np.sin(arg1/2.0)/mag*np.array(arg2))])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        else:
            raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
    
    def __len__(self):
        return len(self.q)
        
    def __repr__(self):
        return "Quaternion: [{}]".format(",".join([str(x) for x in self.q]))
    
    def __getitem__(self, index):
        return self.q[index]


    def __add__(self,other):
        return Quaternion(self.q+other.q)
    
    def __sub__(self,other):
        return Quaternion(self.q-other.q)
    
    def __mul__(self, other):
        # Multiplicación por escalar
        if isinstance(other, (int, float, np.number)):
            return Quaternion(self.q * other)

        # Multiplicación de cuaterniones (igual que @)
        if isinstance(other, Quaternion):
            s1, v1 = self.q[0], self.q[1:]
            s2, v2 = other.q[0], other.q[1:]

            s = s1*s2 - np.dot(v1, v2)
            v = s1*v2 + s2*v1 + np.cross(v1, v2)

            return Quaternion([s, *v])

        # Si llega aquí, es un error
        raise TypeError("Quaternion can only be multiplied by scalar or Quaternion")

    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return 1/other*self

    def __matmul__(self, other):
        # Aseguramos que other sea Quaternion
        if not isinstance(other, Quaternion):
            other = Quaternion(other)

        s1, v1 = self.q[0], self.q[1:4]
        s2, v2 = other.q[0], other.q[1:4]

        s = s1*s2 - np.dot(v1, v2)
        v = s1*v2 + s2*v1 + np.cross(v1, v2)

        return Quaternion([s, *v])

    
    def inverted(self):
        # Para cuaterniones unitarios, el inverso es el conjugado
        return self.conjugated()

    
    def conjugated(self):
        return Quaternion([self[0], -self[1], -self[2], -self[3]])

    def normalized(self):
        mag = self.magnitude()
        return Quaternion(self.q/mag)
    
    def invert(self):
        self.conjugate()

    
    def conjugate(self):
        self.q = np.array([self[0], -self[1], -self[2], -self[3]])

    def normalize(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q = self.q/mag
        
    def magnitude(self):
        return np.linalg.norm(self.q)

    def rotate(self, u):
        v = self @ Quaternion(u) @ self.conjugated()
        return v.q[1:4]   # devolver vector 3D completo

    
def quaternion_to_dcm(q):
    """
    Convierte un cuaternión unitario a una matriz de rotación 3x3 (DCM).
    El cuaternión puede ser de tipo Quaternion (nuestra clase) o un array/lista de 4 elementos.

    Fórmula tomada del documento sección 2.9.3
    """

    # Aseguramos que q sea un array numpy de longitud 4
    if isinstance(q, Quaternion):
        q = q.q
    else:
        q = np.array(q)

    q0, q1, q2, q3 = q  # real, x, y, z

    # Matriz de rotación (Direction Cosine Matrix)
    R = np.array([
        [ q0*q0 + q1*q1 - q2*q2 - q3*q3,   2*(q1*q2 + q0*q3),           2*(q1*q3 - q0*q2) ],
        [ 2*(q1*q2 - q0*q3),               q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 + q0*q1) ],
        [ 2*(q1*q3 + q0*q2),               2*(q2*q3 - q0*q1),           q0*q0 - q1*q1 - q2*q2 + q3*q3 ]
    ])

    return R

def axis_angle_to_dcm(theta, u):
    """
    Construye una matriz de rotación (DCM) a partir de un eje unitario u y un ángulo theta.
    Implementa la fórmula de Rodrigues:
        R = I + sin(theta) * S(u) + (1 - cos(theta)) * S(u)^2
    donde S(u) es la matriz skew-symmetric del vector u.
    """

    u = np.array(u, dtype=float)

    # Normalizamos el eje por seguridad
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-9:
        raise ValueError("El eje de rotación tiene norma cero.")
    u = u / norm_u

    ux, uy, uz = u

    # Matriz skew-symmetric S(u)
    S = np.array([
        [ 0,   -uz,   uy],
        [ uz,   0,   -ux],
        [-uy,  ux,    0 ]
    ])

    I = np.eye(3)

    # Fórmula de Rodrigues
    R = I + np.sin(theta) * S + (1 - np.cos(theta)) * (S @ S)

    return R

def dcm_to_quaternion(R):
    """
    Convierte una matriz de rotación 3x3 (DCM) a un cuaternión unitario.
    Implementa el algoritmo de Shepperd, que es numéricamente estable.

    R debe ser una matriz ortonormal (perteneciente a SO(3)).
    """

    R = np.array(R, dtype=float)

    # Trazar la matriz
    trace = R[0,0] + R[1,1] + R[2,2]

    # Caso 1: traza positiva (más estable)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2,1] - R[1,2]) * s
        q2 = (R[0,2] - R[2,0]) * s
        q3 = (R[1,0] - R[0,1]) * s

    # Caso 2: el mayor elemento está en R[0,0]
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        q0 = (R[2,1] - R[1,2]) / s
        q1 = 0.25 * s
        q2 = (R[0,1] + R[1,0]) / s
        q3 = (R[0,2] + R[2,0]) / s

    # Caso 3: el mayor elemento está en R[1,1]
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        q0 = (R[0,2] - R[2,0]) / s
        q1 = (R[0,1] + R[1,0]) / s
        q2 = 0.25 * s
        q3 = (R[1,2] + R[2,1]) / s

    # Caso 4: el mayor elemento está en R[2,2]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        q0 = (R[1,0] - R[0,1]) / s
        q1 = (R[0,2] + R[2,0]) / s
        q2 = (R[1,2] + R[2,1]) / s
        q3 = 0.25 * s

    q = np.array([q0, q1, q2, q3])

    # Normalizamos por seguridad
    q = q / np.linalg.norm(q)

    return q

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convierte ángulos de Euler (roll, pitch, yaw) a un cuaternión unitario.
    Convención: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)

    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)

    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    # Fórmula estándar para roll-pitch-yaw → quaternion
    q0 = cy*cp*cr + sy*sp*sr
    q1 = cy*cp*sr - sy*sp*cr
    q2 = cy*sp*cr + sy*cp*sr
    q3 = sy*cp*cr - cy*sp*sr

    q = np.array([q0, q1, q2, q3])

    # Normalizamos por seguridad
    q = q / np.linalg.norm(q)

    return q

def quaternion_to_euler(q):
    """
    Convierte un cuaternión unitario a ángulos de Euler (roll, pitch, yaw).
    Convención: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Devuelve (roll, pitch, yaw).
    """

    # Aseguramos que q sea un array numpy
    if isinstance(q, Quaternion):
        q = q.q
    else:
        q = np.array(q)

    q0, q1, q2, q3 = q  # real, x, y, z

    # roll (rotación sobre x)
    sinr = 2.0 * (q0*q1 + q2*q3)
    cosr = 1.0 - 2.0 * (q1*q1 + q2*q2)
    roll = np.arctan2(sinr, cosr)

    # pitch (rotación sobre y)
    sinp = 2.0 * (q0*q2 - q3*q1)
    # clamp por seguridad numérica
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (rotación sobre z)
    siny = 2.0 * (q0*q3 + q1*q2)
    cosy = 1.0 - 2.0 * (q2*q2 + q3*q3)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw

def dcm_to_euler(R):
    """
    Convierte una matriz de rotación 3x3 (DCM) a ángulos de Euler (roll, pitch, yaw).
    Convención: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Devuelve (roll, pitch, yaw).
    """

    R = np.array(R, dtype=float)

    # pitch = -arcsin(R13)
    pitch = np.arcsin(-R[0,2])

    # roll = atan2(R23, R33)
    roll = np.arctan2(R[1,2], R[2,2])

    # yaw = atan2(R12, R11)
    yaw = np.arctan2(R[0,1], R[0,0])

    return roll, pitch, yaw

def euler_to_dcm(roll, pitch, yaw):
    """
    Convierte ángulos de Euler (roll, pitch, yaw) a una matriz de rotación 3x3 (DCM).
    Convención: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Construimos la matriz de rotación directamente
    R = np.array([
        [ cy*cp,              cy*sp*sr - sy*cr,     cy*sp*cr + sy*sr ],
        [ sy*cp,              sy*sp*sr + cy*cr,     sy*sp*cr - cy*sr ],
        [ -sp,                cp*sr,                cp*cr            ]
    ])

    return R


def read_TLE_file(file_name,satellite_name=''):
  def validate_entry(Name,line1,line2):
    if not Name[0].isalpha():
      return False
    if not line1[0].startswith("1") or not len(line1) == 9:
      return False
    if not line2[0].startswith("2") or not len(line2) == 8:
      return False
    return True

  tle_data = []
  with open(file_name) as f:
    file_contents = f.readlines()
  if len(file_contents) < 3:
    print("Error reading file\nRequired format is:\nAAAAAAAAAAAAAAAAAAAAAAAA\n1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN\n2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN\nfor each entry")
    return tle_data

  for i in range(0,len(file_contents),3):
    if(satellite_name in file_contents[i]):
      Name = file_contents[i].strip()
      line1 = file_contents[i+1].strip().split()
      line2 = file_contents[i+2].strip().split()
      if validate_entry(Name,line1,line2):
        epoch = float(line1[3])
        e = float("0."+line2[4])
        rev = float(line2[7])
        Me = float(line2[6])
        i = float(line2[2])
        O = float(line2[3])
        w = float(line2[5])
        tle_data.append((Name,epoch,e,rev,Me,i,O,w))
      else:
        print("Error reading entry:\n{}{}{}".format(file_contents[i],file_contents[i+1],file_contents[i+2]))
        break
  return tle_data

def read_obj(fname):
    verts = []
    vcols = []
    faces = []
    with open(fname,'r') as f:
        for line in f:
            if line.startswith('v '):
                d = [float(x) for x in line.split(' ')[1:]]
                verts.append(d[0:3])
                if len(d) > 3:
                    vcols.append(d[3:])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split(' ')[1:]])
            else:
                pass
    return np.array(verts),np.array(vcols),np.array(faces)

def rotscaleloc_to_vispy(pos=None,quat=None,Rot=None,Eul=None,scale=None):
	if quat is not None:
		q = Quat(w=quat[0],x=quat[1],y=quat[2],z=quat[3])
		H = Mat4(q.conjugate().get_matrix())
	elif Rot is not None:
		p = np.array([[0,0,0]]).T
		HT = np.vstack(((np.hstack((Rot,p)),np.array([[0,0,0,1]]))))
		H = Mat4(HT.T)
	elif Eul is not None:
		q = Quat.create_from_euler_angles(Eul[2],Eul[1],Eul[0])
		H = Mat4(q.conjugate().get_matrix())
	else:
		H = Mat4()
	if scale is not None:
		H.scale((scale,scale,scale))
	if pos is not None:
		H.translate(pos)
	return H

def H_to_Rp(H):
    return H.matrix[:3,:3].T,H.matrix[-1][:3]


def log_pos(name, pos):
    base_path = os.path.dirname(os.path.abspath(__file__))   # ruta absoluta del archivo simutils.py
    data_path = os.path.join(base_path, "data")              # carpeta data absoluta
    file_name = os.path.join(
        data_path,
        name + "_" + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt"
    )
    print("logged: " + file_name)
    np.savetxt(file_name, pos)

# Numerical integrations

def step_euler(h, t_k, x_k, f):
    """
    Forward Euler integrator.
    x_{k+1} = x_k + h * f(t_k, x_k)
    """
    return x_k + h * f(t_k, x_k)

def step_leapfrog(h, t_k, x_k, f):
    """
    Leapfrog integrator.
    State x = [r, v] where r and v are 3D vectors.
    """
    r = x_k[:3]
    v = x_k[3:]

    # acceleration at current step
    a_k = f(t_k, x_k)[3:]

    # half-step velocity
    v_half = v + 0.5 * h * a_k

    # update position
    r_next = r + h * v_half

    # compute acceleration at next step
    x_temp = np.hstack((r_next, v_half))
    a_next = f(t_k + h, x_temp)[3:]

    # full-step velocity
    v_next = v_half + 0.5 * h * a_next

    return np.hstack((r_next, v_next))

def step_verlet(h, t_k, x_k, x_k_1, f):
    """
    Verlet integrator.
    x_k     = current position
    x_k_1   = previous position
    f       = returns acceleration in the last 3 components
    """
    # acceleration at current step
    a_k = f(t_k, np.hstack((x_k, np.zeros(3))))[3:]

    # Verlet update
    x_next = 2*x_k - x_k_1 + h*h * a_k

    return x_next

def step_RK4(h, t_k, x_k, f):
    """
    Classical 4th-order Runge-Kutta integrator.
    Computes x_{k+1} from x_k for the ODE: x' = f(t, x)
    """

    k1 = f(t_k, x_k)
    k2 = f(t_k + 0.5*h, x_k + 0.5*h*k1)
    k3 = f(t_k + 0.5*h, x_k + 0.5*h*k2)
    k4 = f(t_k + h,     x_k + h*k3)

    return x_k + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
