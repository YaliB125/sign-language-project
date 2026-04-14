import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 4 
dataset_size = 100   
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    label = input(f"Enter label for class {j}")
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for {label}. press "q" to start.')
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, f'Target: {label}. Press "q" to start!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Starting collection for {label}...")
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(10)
        
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1
    
    print(f"Done with {label}!")
    cv2.destroyAllWindows()
    time.sleep(0.5) 

cap.release()
cv2.destroyAllWindows()