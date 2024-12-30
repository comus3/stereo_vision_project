import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import glob
from mathutils import geometry as pygeo
from mathutils import Vector
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import os
from IPython.display import Image as pythImage, display




def plotDotWorld(camWorldCenterLeft, camWorldCenterRight, objp):
    fig = plt.figure()                                      # cree une Figure vide
    ax = plt.axes(projection='3d')                          # on cree la figure avec des axes 3D (sans le projection c'est une fig 2D)
    
    ax.scatter3D(objp[:,0],objp[:,1],objp[:,2])             # On pose les points 3D de notre objet
    
    x,y,z,d = camWorldCenterLeft                            # les coord homogenes de notre camera gauche
    ax.scatter(x, y, z, c='g', marker='o')                  # on l affiche avec un une boule verte
    
    x2,y2,z2,d2 = camWorldCenterRight                       # les coord homogenes de notre camera droite
    ax.scatter(x2, y2, z2 , c='r', marker='o')              # on l affiche avec un une boule rouge
    
    plt.show() 
    
def crossMat(v):
    # Soit                          V = [[xxx][yyyy][zzzz]]
    v = v[:,0]    # qui donne donc  V = [xxx yyyy zzzz]               
            # Et on  retourne   ([[0000  -zzzz    yyyy]
            #                   [zzzz   0000   -xxxx]
            #                   [-yyyy  xxxx    0000]])

    return np.array([[0,-v[2],v[1]] , [v[2],0,-v[0]] , [-v[1],v[0],0]])


def matFondamental(camLeft,centerRight,camRight):
        # pseudo inverse de la matrice camRight, qu'on multiplie par camLeft (= MatriceIntrinsèque @ MatriceRota)
        # qu'on multiplie ensuite par (camLeft @centerRight) qui représente ...
        # Et on fait finalement le cross product (produit vectoriel) de la matrice colonne resultat
        return np.array(crossMat(camLeft @ centerRight) @ camLeft @ np.linalg.pinv(camRight))


def getImgLine(fname):              
    img = cv.imread(fname)                                      #stocke l'image dans une variable
    red = img[:,:,2]                                            #concrètement ça évite une erreur dans epiLine mais pq ?
    ret, mask = cv.threshold(red,127,255,cv.THRESH_TOZERO)      #vire tout ce qui n'est pas la ligne rouge
    return mask
    

def findEpilines(path, Fondamental):
    epilines = []
    for l in range(26):  # On a 25 images, donc pour chaque image
        if l < 10:
            strp = path + '000' + str(l) + '.png'
        else:
            strp = path + '00' + str(l) + '.png'

        ImgLine = getImgLine(strp)  # Transform the image into a background with just the line
        pointsLeft = [[], [], []]

        for index, line in enumerate(ImgLine):  # Process each line
            for pixel in line:  # For each pixel in the line
                if pixel != 0:  # If pixel is non-zero (i.e., part of the line)
                    pixel = 1  # Mark it as part of the line (binary)
            try:
                pointsLeft[0].append(np.average(range(1920), weights=line))  # Weighted average for x-coordinate
                pointsLeft[1].append(index)  # y-coordinate (index of the line)
                pointsLeft[2].append(1)  # Homogeneous coordinate
            except:
                pass

        # Compute the epilines in the right image (using the Fundamental matrix)
        epilinesRight = Fondamental @ pointsLeft
        epilines.append([pointsLeft, epilinesRight])  # Store pointsLeft and epilinesRight for each image

    return epilines


def lineY(coef, x):
    """
    Calcule la coordonnée Y sur une ligne donnée les coefficients.
    """
    a, b, c = coef
    return -(c + a * x) / b


def drawAvgPoint(img, EplLeft):
    """
    Ajoute des points colorés sur une image à partir des coordonnées fournies.
    """
    i = 0
    while i < len(EplLeft[0]):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img = cv.circle(img, (int(EplLeft[0][i]), int(EplLeft[1][i])), 5, color, -1)
        i += 10
    return img



def drawEpl(img, EplRight):
    """
    Ajoute des lignes épipolaires directement sur l'image en fonction des coefficients fournis.
    :param img: Image sur laquelle dessiner les lignes.
    :param EplRight: Matrice contenant les coefficients des lignes épipolaires.
    """
    coef, length = EplRight.shape
    for i in range(0, length, 10):  # Espacement des lignes
        # Calculer les points de la ligne
        y1 = int(lineY(EplRight[:, i], 0))       # Point à x=0
        y2 = int(lineY(EplRight[:, i], img.shape[1] - 1))  # Point à x=largeur de l'image

        # Dessiner la ligne sur l'image
        color = (0, 0, 255)  # Couleur rouge (BGR)
        thickness = 2  # Épaisseur de la ligne
        cv.line(img, (0, y1), (img.shape[1] - 1, y2), color, thickness)

    return img



def process_folder(left_folder, right_folder, EplLeft, EplRight, output_gif_path, duration=300):
    """
    Parcourt un dossier d'images, applique drawAvgPoint et drawEpl sur chaque image, 
    et génère un GIF.
    :param left_folder: Dossier contenant les images pour drawAvgPoint.
    :param right_folder: Dossier contenant les images pour drawEpl.
    :param EplLeft: Liste des points pour drawAvgPoint.
    :param EplRight: Liste des coefficients pour drawEpl.
    :param output_gif_path: Chemin du fichier GIF de sortie.
    :param duration: Durée de chaque frame en millisecondes.
    """
    left_images = sorted([f for f in os.listdir(left_folder) if f.endswith('.png')])
    right_images = sorted([f for f in os.listdir(right_folder) if f.endswith('.png')])

    frames = []  # Liste pour stocker les frames

    for idx in range(len(left_images)):
        # Charger les images
        left_path = os.path.join(left_folder, left_images[idx])
        right_path = os.path.join(right_folder, right_images[idx])

        img_left = cv.imread(left_path)
        img_right = cv.imread(right_path)

        # Appliquer les transformations
        img_left = drawAvgPoint(img_left, EplLeft[idx][0])
        img_right = drawEpl(img_right, EplRight[idx][1])

        # Combiner les deux images côte à côte
        combined_width = img_left.shape[1] + img_right.shape[1]
        combined_height = max(img_left.shape[0], img_right.shape[0])
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        combined_img[:img_left.shape[0], :img_left.shape[1]] = img_left
        combined_img[:img_right.shape[0], img_left.shape[1]:] = img_right

        # Convertir en format compatible Pillow (RGB)
        combined_img_rgb = cv.cvtColor(combined_img, cv.COLOR_BGR2RGB)
        frames.append(Image.fromarray(combined_img_rgb))

    # Créer le GIF
    if frames:
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"GIF créé : {output_gif_path}")
    else:
        print("Aucune image traitée.")

  
    
def getReddAvg(fname):
    red = getImgLine(fname)
    redPoints = [[],[],[]]

    for i, line in enumerate(red):
        for pixel in line:
            if pixel != 0:
                pixel = 1
        try:
            redPoints[0].append(np.average(range(1920), weights = line))
            redPoints[1].append(i)
            redPoints[2].append(1)
        except:
            pass
    return redPoints


def eplRedPoints(path,EplRight):
    points = []
    for l in range(26):
        if l<10:
            strp = path + '000' + str(l) +'.png'
        else:
            strp = path + '00' + str(l) +'.png'
            
        redPoints = getReddAvg(strp)
        scan = cv.imread(strp)

        pointsRight = [[],[],[]]
        eplImg = EplRight[l][1]
        # print(strp)
        for i in range(len(eplImg[0])):
            try : 
                x = int(redPoints[0][i])
                y = int(lineY(eplImg[:,i],x))
                pointsRight[0].append(x)
                pointsRight[1].append(y)
                pointsRight[2].append(1)
                
                color = tuple(np.random.randint(0,255,3).tolist())
                scan = cv.circle(scan,(x,y),5,color,-1)
            except:
                pass
        points.append(pointsRight)
        # plt.imshow(scan)
        # plt.show()
    return points

def arrayToVector(p):
    return Vector((p[0],p[1],p[2]))


def getIntersection(pointsLeft,pointsRight,camWorldCenterLeft,camWorldCenterRight,camLeft,camRight):
    
    pL = np.array(pointsLeft)
    pR = np.array(pointsRight)
    
    camCenterRight = np.transpose(camWorldCenterRight)[0]
    camCenterLeft = np.transpose(camWorldCenterLeft)[0]
    
    # calcul du point sur l'object en applicant la pseudo-inverse de la camera sur le point trouvé plus-haut
    
    leftObject = (np.linalg.pinv(camLeft) @ pL)
    rightObject = (np.linalg.pinv(camRight) @ pR) 
    
    # conversion des np.array en mathutils.Vector pour l'utilisation de la methode d'intersection
    
    leftEndVec = arrayToVector(leftObject)
    rightEndVec = arrayToVector(rightObject)
    
    leftStartVec = arrayToVector(camCenterLeft)
    rightStartVec = arrayToVector(camCenterRight)
    
    # affichage des lignes reliant centre à point objet
    
    '''
    draw3DLine(camCenterLeft,leftObject)
    draw3DLine(camCenterRight,rightObject)
    plt.show()
    '''
    
    # utilisation de mathutils.geometry.intersect_line_line pour trouver l'intersection des lingnes passant par les 2 
    # points. 
    return pygeo.intersect_line_line(leftStartVec,leftEndVec,rightStartVec,rightEndVec)


def draw3DLine(start,end):
    figure = plt.figure()
    ax = Axes3D(figure)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    x_start,y_start,z_start = start
    x_end,y_end,z_end = end

    print("start = ({},{},{})".format(x_start,y_start,z_start))
    print("end = ({},{},{})\n".format(x_end,y_end,z_end))

    ax.scatter(x_start,y_start,z_start,c='r',marker='o')
    ax.plot([x_start ,x_end],[y_start,y_end],[z_start,z_end])


def getObjectPoint(pointsRight,epl,camWorldCenterLeft,camWorldCenterRight,camLeft,camRight):
    point = [[],[],[]]
    for l in range(26):
        pointsLeft = np.array(epl[l][0])
        
        pointRight = np.array(pointsRight[l])
        for i in range(len(pointsLeft[0])):
            try:
                
                # calcul du point d'intersection sur l'objet -> on obtient une liste de vector
                intersection = getIntersection(pointsLeft[:,i],pointRight[:,i],camWorldCenterLeft,camWorldCenterRight,camLeft,camRight)
                #print(intersection)
                for inter in intersection:
                    inter *= 1000
                    x,y,z = inter
                    point[0].append(x)
                    point[1].append(y)
                    point[2].append(z)
            except:
                pass
    return np.array(point)
        

def drawPointObject(point):
    """
    Dessine un nuage de points 3D à partir des coordonnées données.
    """
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')  # Crée un axe 3D

    # Dessine les points en 3D
    ax.scatter3D(point[0, :], point[1, :], point[2, :], s=100, c='black', marker='x')

    # Ajuste l'angle de vue
    ax.view_init(-90, -70)
    plt.axis('off')  # Supprime les axes
    plt.show()

    


def drawSurfaceObject(point):
    """
    Dessine un objet 3D à partir des points donnés.
    """
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')  # Ajout d'un axe 3D
    ax.plot_trisurf(point[0, :], point[1, :], point[2, :])  # Surface 3D

    ax.view_init(-95, -50)  # Ajuste l'angle de vue
    plt.axis('off')  # Supprime les axes
    plt.show()
    
    
def pointToJson(point):
    data = {'x':point[0,:].tolist(),'y':point[1,:].tolist(),'z':point[2,:].tolist()}
    with open('point.txt','+w') as file:
        json.dump(data,file)


