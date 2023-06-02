import cv2
import mediapipe as mp

# Criando um objeto de desenho
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Criando dois objetos, o FaceMesh e o Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

def doAll(resize=0, file=0, flip=0):
    cap = cv2.VideoCapture(file)
        
    # Configuração do Media Pipe Face Mesh
    face = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5)
    # Configuração do Media Pipe Hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
        
    while cap.isOpened():
        sucess, image = cap.read()
        
        if not sucess:
            continue
            
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if flip == 1:
            image = cv2.flip(image, 1)
        
        # Detecção da face
        results = face.process(image)
        
        # Detecção das mãos
        results_hands = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if resize != 0:
            image = cv2.resize(image, resize)
        
        # Renderizando as marcações da face
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
            
        # Renderizando as marcações das mãos
        if results_hands.multi_hand_landmarks:
            for hand in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand,
                    connections=mp_hands.HAND_CONNECTIONS
                )
            
        cv2.imshow('Modelo ML', image)
            
        if cv2.waitKey(5) & 0xFF == ord(' '):
            break
        
    cap.release()

opção = input('Você quer usar a webcam ou usar um arquivo?\n')[0].upper()

if opção == 'W':
    doAll(flip=1)
    
elif opção == 'A':
    file = input('Caminho: ').replace('\\', '\\\\')
    
    doAll(resize=(800, 500), file=file)