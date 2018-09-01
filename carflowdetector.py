#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualizador():
	def __init__(self, tamano='min'):	#tamano (320,240)
		#Position Variables
		self.w = 320
		self.h = 240
		if tamano == 'max':
			self.w = 640
			self.h = 480
			#print('maximum selected')
		self.size = (self.w,self.h)
		self.centrox = self.w // 2
		self.centroy = self.h // 2
		self.centroFrame = (self.centrox,self.centroy)

		#Auxiliares
		self.font = cv2.FONT_HERSHEY_SIMPLEX

	def draw_vector(self,imagen,vector,color):
		imagen = cv2.line(imagen,(self.centroFrame[0],self.centroFrame[1]),(self.centroFrame[0]+int(vector[0]),self.centroFrame[1]+int(vector[1])),color,2)
		return imagen

	def drawMagnitudAtBottomCorner(self,imagen,magnitud,indice):
		ancho = 4
		offset = 5
		color = (255,255,255)
		if indice==0:
			color = (255,0,0)
		elif indice==1:
			color = (0,255,0)
		elif indice==2:
			color = (0,0,255)
		imagen = cv2.line(imagen,(int(self.w-offset-2*ancho*indice),int(self.h-offset)),(int(self.w-offset-2*ancho*indice),int(self.h-offset-2*magnitud)),color,ancho)
		return imagen

	def overlayRegions(self,imagen,vertices,indice):
		miCapa = imagen.copy()
		if indice==0 :
			cv2.fillPoly(miCapa,[np.array(vertices)],(255,0,0))
		elif indice==1:
			cv2.fillPoly(miCapa,[np.array(vertices)],(0,255,0))
		elif indice==2:
			cv2.fillPoly(miCapa,[np.array(vertices)],(0,0,255))
		else:
			print('Not applied mask!')
			return imagen

		opacity = 0.15
		newImage = cv2.addWeighted(miCapa, opacity, imagen, 1 - opacity, 0, imagen)
		return newImage

	def putLabelAtCorner(self,imagenALabelar, esquina, etiqueta):
		x_pos = 0
		y_pos = 0
		if esquina == 0:
			x_pos = 10
			y_pos = 10
		elif esquina == 1:
			x_pos = self.w//2
			y_pos = 10
		elif esquina == 2:
			x_pos = self.w//2
			y_pos = self.h
		elif esquina == 3:
			x_pos = 0
			y_pos = self.h
		resultado = cv2.putText(imagenALabelar,etiqueta,(x_pos,y_pos), self.font, 0.5, (255,0,0), 2, cv2.LINE_AA)
		return resultado

class DetectorAutomoviles():
	def __init__(self):	#size (320,240)
	# BackGroundSubs
		self.carSize = 800
		self.carSizeMaximum = 6000
		self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		self.back=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

	def getCarCenters(self,current_image):
		fram = cv2.medianBlur(current_image,11) # 7 default
		fgmask = self.back.apply(fram)
		blur = cv2.GaussianBlur(fgmask,(7,7),1)
		fgmask = cv2.morphologyEx(blur, cv2.MORPH_OPEN, self.kernel)
		contor = fgmask.copy()
		im, contours,hierarchy =cv2.findContours(contor,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		centers = []
		rectangles = []
		for cantObjetos in contours:
			if (cv2.contourArea(cantObjetos) >= self.carSize) & (cv2.contourArea(cantObjetos) <= self.carSizeMaximum):
			#if cv2.contourArea(cantObjetos) >=self.carSize:
				x,y,ancho,alto=cv2.boundingRect(cantObjetos)
				a=int(x+ancho/2)
				b=int(y+alto/2)
				centers.append([a,b])
				rectangles.append([x,y,ancho,alto])
		return np.array(centers), np.array(rectangles)


class FlowDetector():
	def __init__(self,angle=55):	#size (320,240)
		#Auxiliar variables
		# La primera imagen de inicializacion puede entrar en escala a color
		self.anchoZebra = 0
		self.largoZebra = 0
		self.sizeZebra = (self.anchoZebra,self.largoZebra)
		self.centrox = 0
		self.centroy = 0
		self.centroFrame = (self.centrox,self.centroy)

		self.momentumMinimoAuto = 20

		self.auxiliar_image = np.array([])

		# Class variables
		self.theta = angle
		self.optimalStep = 8

		# flow sume fx,fy va
		self.vertices = []

		# parámetrosFisicos
		self.velocidades=[]
		self.momentumActual = 0


		# Variables para el filtro:
		self.MagnitudesVelocidad = np.array((0.0,0.0,0.0,0.0))
		self.MagnitudesVelocidadFiltradas = np.array((0.0,0.0,0.0,0.0))
		self.velocidadesFiltradas = []
		self.a_coeff = np.array(( 1.,-2.37409474,1.92935567,-0.53207537))
		self.b_coeff = np.array(( 0.00289819,0.00869458,0.00869458,0.00289819))
		
		# Deteccion de automoviles
		self.minimoValorVelocidad = 4
		self.velocidadesSinNegativo = []
		self.pulsosAutomoviles_funcionSigno = []
		self.minimoMomentum = 20
		self.indiceActual = 0

		#Auxiliares
		self.font = cv2.FONT_HERSHEY_SIMPLEX

	def borrarRegistros(self):
		self.velocidades=[]
		self.velocidadesFiltradas = []
		self.velocidadesSinNegativo = []
		self.pulsosAutomoviles_funcionSigno = []


	def inicializarClase(self,imagen):
		# La primera imagen de inicializacion puede entrar en escala a color
		self.anchoZebra = imagen.shape[1]
		self.largoZebra = imagen.shape[0]
		self.sizeZebra = (self.anchoZebra,self.largoZebra)
		self.centrox = self.anchoZebra // 2
		self.centroy = self.largoZebra // 2
		self.centroFrame = (self.centrox,self.centroy)

		self.auxiliar_image = self.introduce_image(imagen)

	def introduce_image(self,imagen):
		try:
			imagen = cv2.cvtColor(np.array(imagen), cv2.COLOR_BGR2GRAY)
		except:
			print('Chequea la imagen dude')
		return imagen

	# Unicamente alteramos la imagen como auxiliar, con los estándares de normalización
	def set_previousImage(self,previousImage):
		self.auxiliar_image = self.introduce_image(previousImage)
		return self.auxiliar_image

	def draw_vector(self,imagen,vector,color):
		cv2.line(imagen,(self.centroFrame[0],self.centroFrame[1]),(self.centroFrame[0]+int(vector[0]),self.centroFrame[1]+int(vector[1])),color,2)
		return imagen

	def procesarNuevoFrame(self, current_image):
		y, x = np.mgrid[self.optimalStep/2:self.largoZebra:self.optimalStep, self.optimalStep/2:self.anchoZebra:self.optimalStep].reshape(2,-1)
		y = np.int32(y)
		x = np.int32(x)
		#flow = cv2.calcOpticalFlowFarneback(self.auxiliar_image, current_image, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags, flow)
		current_image = self.introduce_image(current_image)
		flow = cv2.calcOpticalFlowFarneback(self.auxiliar_image, current_image, None, 0.5, 3, 15, 3, 5, 1.2, 0) #(self.auxiliar_image, current_image, None, 0.7, 3, 9, 3, 5, 1.2, 0)
		fx, fy = flow[y,x].T
		total_flow_framex = sum(fx)
		total_flow_framey = sum(fy)
		total_flow = np.array([total_flow_framex, total_flow_framey])
		module = np.sqrt(total_flow[0]**2  + total_flow[1]**2)
		lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
		lines = np.int32(lines + 0.5)
		return total_flow, module, lines
		
	def procesarFlujoEnTiempoReal(self,vector):
		unitary_vector = np.array([-1,0])
		scalar_vel = vector[0]*unitary_vector[0] + vector[1]*unitary_vector[1]
		vector_vel = scalar_vel * unitary_vector
		module_vector_vel = np.sqrt(vector_vel[0]**2 + vector_vel[1]**2)

		#Filtrando vector
		self.MagnitudesVelocidad[3] = self.MagnitudesVelocidad[2]
		self.MagnitudesVelocidad[2] = self.MagnitudesVelocidad[1]
		self.MagnitudesVelocidad[1] = self.MagnitudesVelocidad[0]
		self.MagnitudesVelocidad[0] = scalar_vel
		
		self.MagnitudesVelocidadFiltradas[3] = self.MagnitudesVelocidadFiltradas[2]
		self.MagnitudesVelocidadFiltradas[2] = self.MagnitudesVelocidadFiltradas[1]
		self.MagnitudesVelocidadFiltradas[1] = self.MagnitudesVelocidadFiltradas[0]
		self.MagnitudesVelocidadFiltradas[0] = - self.a_coeff[1]*self.MagnitudesVelocidadFiltradas[1]-self.a_coeff[2]*self.MagnitudesVelocidadFiltradas[2]-self.a_coeff[3]*self.MagnitudesVelocidadFiltradas[3]+self.b_coeff[0]*self.MagnitudesVelocidad[0]+self.b_coeff[1]*self.MagnitudesVelocidad[1]+self.b_coeff[2]*self.MagnitudesVelocidad[2]+self.b_coeff[3]*self.MagnitudesVelocidad[3]
		
		smooth_vector = self.MagnitudesVelocidadFiltradas[0]*unitary_vector

		velocidadReal = self.MagnitudesVelocidadFiltradas[0]*7.2/100 #km/h
		self.velocidades.append(self.MagnitudesVelocidad[0])
		self.velocidadesFiltradas.append(self.MagnitudesVelocidadFiltradas[0])
		if self.MagnitudesVelocidadFiltradas[0] <= self.minimoValorVelocidad:
			self.velocidadesSinNegativo.append(0)
		else:
			self.velocidadesSinNegativo.append(self.MagnitudesVelocidadFiltradas[0])
		
		self.pulsosAutomoviles_funcionSigno.append(100*np.sign(self.velocidadesSinNegativo[len(self.velocidadesSinNegativo)-1]-self.velocidadesSinNegativo[len(self.velocidadesSinNegativo)-2]))
		self.momentumActual += self.velocidadesSinNegativo[len(self.velocidadesSinNegativo)-1]
		flancoDeSubida = self.pulsosAutomoviles_funcionSigno[len(self.pulsosAutomoviles_funcionSigno)-1]-self.pulsosAutomoviles_funcionSigno[len(self.pulsosAutomoviles_funcionSigno)-2]
		self.indiceActual = 0
		# En el flanco de subida se resetea el momentum
		if (flancoDeSubida > 0)&(self.pulsosAutomoviles_funcionSigno[len(self.pulsosAutomoviles_funcionSigno)-1]==100):
			self.indiceActual = 1
		# En el flanco de bajada se resetea el momentum a -1 simbolizando que no hay automovil

		if (flancoDeSubida > 0):
			self.momentumActual = 0

		if (flancoDeSubida < 0):
			self.indiceActual = -1

		return self.momentumActual, smooth_vector, velocidadReal, self.indiceActual

	def obtenerPulsoAutomoviles(self):
		return self.pulsosAutomoviles_funcionSigno[len(self.pulsosAutomoviles_funcionSigno)-1]

	def obtenerVelocidadSinNegativo(self):
		return self.velocidadesSinNegativo[len(self.velocidadesSinNegativo)-1]


class Stabilizador():
	def __init__(self,initial_image, rectangulo_a_seguir):	# [[x0,y0],[x1,y1]]
		self.rectanguloX0 = rectangulo_a_seguir[0][0]
		self.rectanguloY0 = rectangulo_a_seguir[0][1]
		self.rectanguloX0 = rectangulo_a_seguir[1][0]
		self.rectanguloY1 = rectangulo_a_seguir[1][1]
		self.imagen_auxiliar_croped = initial_image[rectanguloY0:rectanguloY1,rectanguloX0:rectanguloX1]
		self.imagen_auxiliar_croped = cv2.cvtColor(np.array(self.imagen_auxiliar_croped), cv2.COLOR_BGR2GRAY)
		self.feature_parameters = dict(maxCorners = 4, qualityLevel = 0.3, minDistance = 7, blockSize= 7)
		self.lk_parameters = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
		self.colour = np.random.randint(0,255,(4,3))	# For four corners for 3 colours RGB between 0 and 255
		self.puntos_to_track = cv2.goodFeaturesToTrack(self.imagen_auxiliar_croped,mask=None,**self.feature_parameters)
		self.mask = np.zeros_like(self.imagen_auxiliar_croped)

	def obtener_vector_desplazamiento(self,nueva_Imagen):
		imagen_actual_croped = nueva_Imagen[rectanguloY0:rectanguloY1,rectanguloX0:rectanguloX1]
		imagen_actual_croped = cv2.cvtColor(np.array(imagen_actual_croped), cv2.COLOR_BGR2GRAY)
		puntos_tracked, st,err = cv2.calcOpticalFlowPyrLK(self.imagen_auxiliar_croped,imagen_actual_croped,self.puntos_to_track,**self.lk_parameters)
		good_new = puntos_tracked[st==1]
		good_old = puntos_to_track[st==1]
		vector_desplazamiento = (0,0)
		for i, (new,old) in enumerate(zip(good_new,good_old)):	# para cada punto obtienes las posiciones iniciales y finales
			a,b = new.ravel()
			c,d = old.ravel()
			vector_desplazamiento +=(c-a,d-b)
			#mask = cv2.line(mask,(a,b),(c,d),self.colour[i].tolist(),2)
			#frame = cv2.circle(imagen_actual_croped,(a,b),5,self.colour[i].tolist(),-1)
		vector_desplazamiento = vector_desplazamiento//4
		visualizacion = cv2.add(frame,mask)
		return visualizacion, vector_desplazamiento

	def estabilizar_imagen(self,imagen_a_estabilizar):
		filas,columnas = imagen_a_estabilizar.shape[:2]
		vector_a_desplazar = self.obtener_vector_desplazamiento(imagen_a_estabilizar)
		matriz_de_traslacion = np.float([[1,0,-vector_a_desplazar[0]],[0,1,-vector_a_desplazar[y]]])
		imagen_estabilizada = cv2.warpAffine(imagen_a_estabilizar,matriz_de_traslacion,(columnas,filas))
		return imagen_estabilizada
