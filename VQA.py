from qiskit import QuantumCircuit, execute, Aer
import numpy as np
from math import pi
from itertools import cycle
from scipy.optimize import minimize


backend = Aer.get_backend('unitary_simulator')

#Single qubit rotations
def rx(theta):
    rx=np.array([[np.cos(theta),-np.sin(theta)*1j],
                 [-np.sin(theta)*1j,np.cos(theta)]])
    return rx
 
def ry(theta):
    ry=np.array([[np.cos(theta),-np.sin(theta)],
                 [np.sin(theta),np.cos(theta)]])
    return ry 

def rz(theta):
    rz=np.array([[np.exp(-1j*theta),0],
                 [0,np.exp(1j*theta)]])
    return rz   
#identity
I=np.identity(2)


#Simplified Hardware eficient ansatz (shea)
def shea(n,p,d,entangs):
    ns=2**n
    pc=cycle(p)
    shea=np.identity(ns)
    for dep in range(d):
        rl1=1
        rl2=1
        rl3=1
        for qn in range(n):
            rl1=np.kron(rl1,rz(next(pc)))
            rl2=np.kron(rl2,ry(next(pc)))
            rl3=np.kron(rl3,rz(next(pc)))
    
        shea=entangs.dot(rl3).dot(rl2).dot(rl1).dot(shea)       
    return shea

#Hardware eficient ansatz (hea)
def hea(n,p,d,entangs):
    ns=2**n
    pc=cycle(p)
    hea=np.identity(ns)

    for dep in range(d):
        rl1=1
        rl2=1
        rl3=1
    
        for qn in range(n):
            rl1=np.kron(rl1,rz(next(pc)))
            rl2=np.kron(rl2,ry(next(pc)))
            rl3=np.kron(rl3,rz(next(pc)))
            
        #The controlled Y rotations are expressed as a combination of two CNOT ant 2 Y rotations
        crot=[] #controlled rotations from the 2 qubit hates
        for r_ind in range(n):#where to put the rotation
            ang=next(pc) #this is the angle for the controlled y rotation
            rot_lp=1
            for ind in range(n):
                if ind==r_ind:
                    rot_lp=np.kron(rot_lp,ry(ang))
                else:
                    rot_lp=np.kron(rot_lp,I)
            crot.append(rot_lp) 
            
            rot_ln=1
            for ind in range(n):
                if ind==r_ind:
                    rot_ln=np.kron(rot_ln,ry(-ang))
                else:
                    rot_ln=np.kron(rot_ln,I)
            crot.append(rot_ln) 
            
        crys=np.identity(ns) #multiplication of all the controlled y torations        
        for cind in range(n):
            crys=crot[2*cind].dot(entangs[cind]).dot(crot[2*cind+1]).dot(entangs[cind]).dot(crys)

        hea=crys.dot(rl3).dot(rl2).dot(rl1).dot(hea)     
    return hea

#Entangling part of the ansatz
def entanglers (n,ansatz): 
    if ansatz=="shea" :
        ent=QuantumCircuit(n)
        for qn in range(n-1):
            ent.cx(qn+1,qn)
        ent.cx(0,n-1)
        job = execute(ent, backend)
        result = job.result()
        entanglers=result.get_unitary(ent)
        
    elif ansatz=="hea": 
        entanglers=[]
        for qn in range(n-1):  
            enta=QuantumCircuit(n)
            enta.cx(n-1-qn,n-2-qn)
            job = execute(enta, backend)
            result = job.result()
            entam=result.get_unitary(enta)
            entanglers.append(entam)
        
        enta=QuantumCircuit(n)
        enta.cx(0,n-1)
        job = execute(enta, backend)
        result = job.result()
        entam=result.get_unitary(enta)
        entanglers.append(entam) 
    return entanglers

#K-contolled NOT gate
def cnx (n):
    ns=2**n
    cnx = np.identity(ns)
    cnx[np.ix_([ns-2,ns-1],[ns-2,ns-1])]=np.array([[0,1],[1,0]])
    return cnx

#Cost function, Hilbert-Schmidth product
def F (p,ansatz,n,d,entangs,target_op):
    if ansatz=="shea":
        U=shea(n,p,d,entangs)[:]
    elif ansatz=="hea":
        U=hea(n,p,d,entangs)[:]
    F=-np.trace(target_op.conj().T.dot(U)) 

    return F