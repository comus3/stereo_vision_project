from tqdm import tqdm
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
import imageio
from scipy.spatial import Delaunay
import os



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


def matFondamental(camLeft,camCenterLeft,camRight):
        # VALIDE
        # formule de l'énoncé pour trouver la matrice fondamentale
        return np.array(crossMat(camRight @ camCenterLeft) @ camRight @ np.linalg.pinv(camLeft))


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



def drawEpl(img, EplRight, spacing=10):
    """
    Ajoute des lignes épipolaires directement sur l'image en fonction des coefficients fournis,
    avec un espacement défini entre les lignes pour éviter qu'il y en ait trop.
    :param img: Image sur laquelle dessiner les lignes.
    :param EplRight: Matrice contenant les coefficients des lignes épipolaires (a, b, c).
    :param spacing: Espacement entre les lignes épipolaires à dessiner.
    """
    # Dimensions de l'image
    xmax = img.shape[1]  # Largeur de l'image
    ymax = img.shape[0]  # Hauteur de l'image
    
    # Dessiner les lignes épipolaires avec l'espacement
    for i in range(0, EplRight.shape[1], spacing):  # Espacement entre les lignes
        # Coefficients de la ligne épipolaire
        a, b, c = EplRight[:, i]
        
        # Calcul du premier point d'intersection (yr1) à x=0
        yr1 = -c / b
        
        # Calcul du deuxième point d'intersection (yr2) à x=xmax
        yr2 = (-xmax * a - c) / b
        
        # Clamper les valeurs pour qu'elles soient dans les limites de l'image
        yr1 = max(0, min(yr1, ymax - 1))
        yr2 = max(0, min(yr2, ymax - 1))

        # Dessiner la ligne entre (0, yr1) et (xmax, yr2)
        color = (0, 0, 255)  # Couleur rouge (BGR)
        thickness = 2  # Épaisseur de la ligne
        cv.line(img, (0, int(yr1)), (xmax - 1, int(yr2)), color, thickness)

    return img




def process_folder(left_folder, right_folder, EplPointsLeft, EplRight, output_gif_path, duration=300):
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
        img_left = drawAvgPoint(img_left, EplPointsLeft[idx])
        img_right = drawEpl(img_right, EplRight[idx])

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
  
    return pygeo.intersect_line_line(leftStartVec,leftEndVec,rightStartVec,rightEndVec)



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
        

def drawPointObject(point, save_path=None,thickness=50):
    """
    Dessine un nuage de points 3D à partir des coordonnées données,
    avec une coloration basée sur la profondeur (coordonnée Z).
    """
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')  # Crée un axe 3D

    # Calcul de la profondeur pour la coloration
    depth = point[2, :]  # Coordonnée Z des points
    colors = plt.cm.jet((depth - np.min(depth)) / (np.max(depth) - np.min(depth)))  # Normalisation et application d'une colormap

    # Dessine les points en 3D
    scatter = ax.scatter3D(point[0, :], point[1, :], point[2, :], s=thickness, c=colors, marker='o')

    # Ajuste l'angle de vue
    ax.view_init(elev=-90, azim=-70)

    # Supprime les axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.axis('off')
        plt.show()
        
def rotatePoints(points, axis='z', angle_deg=180):
    """
    Applique une rotation de angle_deg degrés autour de l'axe spécifié ('x', 'y' ou 'z') sur un ensemble de points.

    Parameters:
        points (np.ndarray): Tableau de points 3D de forme (3, N).
        axis (str): Axe de rotation ('x', 'y' ou 'z').
        angle_deg (float): Angle de rotation en degrés.

    Returns:
        np.ndarray: Points après la rotation.
    """
    angle_rad = np.radians(angle_deg)  # Convertir l'angle en radians
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axe invalide. Choisissez 'x', 'y' ou 'z'.")

    # Appliquer la rotation à tous les points
    return np.dot(rotation_matrix, points)

def createGifMonkey(points, num_frames=15, axis='z', angle_step=10, gif_path='rotation.gif',point_thickness=50):
    """
    Crée un GIF montrant une rotation du nuage de points.
    """
    filenames = []
    for i in range(num_frames):
        rotated_points = rotatePoints(points, axis=axis, angle_deg=i * angle_step)
        filename = f'frame_{i:02d}.png'
        drawPointObject(rotated_points, save_path=filename,thickness=point_thickness)
        filenames.append(filename)

    # Créer le GIF avec boucle infinie
    with imageio.get_writer(gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Supprimer les images temporaires
    for filename in filenames:
        os.remove(filename)

    return gif_path


def drawMeshObject(point, save_path=None):
    """
    Dessine une surface lisse approximée à partir d'un nuage de points 3D
    avec une triangulation de Delaunay, et applique une coloration basée sur la profondeur.
    """
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')  # Crée un axe 3D

    # Triangulation des points avec Delaunay
    tri = Delaunay(point[:2, :].T)  # Utilisation des 2 premières coordonnées pour la triangulation 2D

    # Calcul de la profondeur pour la coloration
    depth = point[2, :]  # Coordonnée Z des points
    colors = plt.cm.jet((depth - np.min(depth)) / (np.max(depth) - np.min(depth)))  # Normalisation et application d'une colormap

    # Dessine les triangles de la surface
    for simplex in tqdm(tri.simplices, desc='Dessin des triangles', unit='triangle', colour='green'):
        tri_points = point[:, simplex]
        ax.plot_trisurf(tri_points[0], tri_points[1], tri_points[2], color=colors[simplex[0]], linewidth=0.5, edgecolor='k', alpha=0.6)

    # Ajuste l'angle de vue
    ax.view_init(elev=-90, azim=-70)

    # Supprime les axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.axis('off')
        plt.show()

def createMaskGifMonkey(points, num_frames=15, axis='z', angle_step=10, gif_path='mask_rotation.gif'):
    """
    Crée un GIF montrant une rotation d'une surface lisse approximée à partir du nuage de points.
    """
    
    filenames = []
    for i in tqdm(range(num_frames)):
        rotated_points = rotatePoints(points, axis=axis, angle_deg=i * angle_step)
        filename = f'frame_{i:02d}.png'
        drawMeshObject(rotated_points, save_path=filename)
        filenames.append(filename)

    # Créer le GIF avec boucle infinie
    with imageio.get_writer(gif_path, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Supprimer les images temporaires
    for filename in filenames:
        os.remove(filename)

    return gif_path


def getObjectPointCV(pointsRightInput, epl, camLeft, camRight):
    """
    Compute 3D points from a sequence of 2D points across 26 images.

    Parameters:
        pointsRightInput (list): List of 2D points in the right images (26 elements, each Nx2).
        epl (list): List of 2D epipolar line points in the left images (26 elements, each Nx2).
        camLeft (np.ndarray): 3x4 projection matrix of the left camera.
        camRight (np.ndarray): 3x4 projection matrix of the right camera.

    Returns:
        np.ndarray: 3xN array of 3D points in Cartesian coordinates.
    """
    # Initialize lists for collecting 3D points
    points_3D_list = [[], [], []]

    # Iterate through all 26 scans
    for l in range(26):
        # Extract points for the left and right images (Nx2)
        pointsLeft = np.array(epl[l]).T[:, :2]  # Ensure Nx2 shape
        pointsRight = np.array(pointsRightInput[l]).T[:, :2]  # Ensure Nx2 shape

        # If there are no points, skip this iteration
        if pointsLeft.size == 0 or pointsRight.size == 0 or pointsLeft.shape[0] != pointsRight.shape[0]:
            continue  # Skip if no points or mismatched number of points

        # Triangulate 3D points
        points_4D = cv.triangulatePoints(camLeft, camRight, pointsLeft.T, pointsRight.T)

        # Normalize to Cartesian coordinates, avoiding division by zero
        w = points_4D[3]
        valid_mask = w != 0  # Mask out points with zero 'w' values
        
        if valid_mask.any():
            # Normalize only valid points
            points_3D = points_4D[:3, valid_mask] / w[valid_mask]  # Normalize

            # Check for invalid points (NaN, Inf)
            points_3D = np.nan_to_num(points_3D, nan=0.0, posinf=0.0, neginf=0.0)

            # Append the valid 3D points to the main list
            points_3D_list[0].extend(points_3D[0])
            points_3D_list[1].extend(points_3D[1])
            points_3D_list[2].extend(points_3D[2])

    # Convert to a numpy array and return
    points_3D_array = np.array(points_3D_list)
    return points_3D_array
