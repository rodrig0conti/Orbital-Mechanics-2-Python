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
    
    def __getitem__(self,index):
        if type(index) == slice:
            if index.stop < index.start:
                raise IndexError("starting index should be smaller than ending index")
            elif index.start in range(0,len(self)+1) and index.stop in range(0,len(self)+1):
                return np.array([self[i] for i in range(index.start,index.stop+1)])
            else:
                raise IndexError("Indexes out of bounds")
        else:
            if index > 3:
                raise IndexError("Index out of bounds")
            else:
                return self.q[index]

    def __add__(self,other):
        return Quaternion(self.q+other.q)
    
    def __sub__(self,other):
        return Quaternion(self.q-other.q)
    
    def __mul__(self,other):
        return Quaternion(self.q*other)
            
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return 1/other*self

    def __matmul__(self,other):
        return Quaternion([self[0]*other[0]-np.dot(self[1:3],other[1:3]), *(self[0]*other[1:3]+other[0]*self[1:3]+np.cross(self[1:3],other[1:3]))])
    
    def inverted(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        return 1.0/mag**2.0*self.conjugated()
    
    def conjugated(self):
        return Quaternion([self[0],*(-self[1:3])])

    def normalized(self):
        mag = self.magnitude()
        return Quaternion(self.q/mag)
    
    def invert(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q /= mag**2.0
    
    def conjugate(self):
        self.q = np.array([self[0],*(-self[1:3])])
    
    def normalize(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q = self.q/mag
        
    def magnitude(self):
        return np.linalg.norm(self.q)

    def rotate(self, u):
        v = self@Quaternion(u)@self.conjugated()
        return v[1:3]

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


