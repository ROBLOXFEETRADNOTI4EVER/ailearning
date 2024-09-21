import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import random
from time import sleep

#load your trained yolov8 model
model = YOLO("/home/blueberry/Desktop/learning/rock-paper-scissors-14/runs/detect/train2/weights/last.pt")

cap = cv2.VideoCapture(0)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def play_rock_paper_scissors(player_choice):
    
    moves = ["Rock","Paper","Scissors"]
    computer_choice = random.choice(moves) #Randomly selects from moves 

    #checking who will win the game

    if player_choice == computer_choice:
        return f"Draw! Both chose {player_choice}."#if it's a draw
    elif (player_choice == "Rock" and computer_choice == "Scissors") or \
         (player_choice == "Paper" and computer_choice == "Rock") or \
         (player_choice == "Scissors" and computer_choice == "Paper"):
        return f"You win! {player_choice} beats {computer_choice}."
    else:
        return f"You lose! {computer_choice} beats {player_choice}."

def countdown():
    for countddown in range(3, 0, -1):
        print(countddown)
        sleep(1)
    print("Rock", "Paper","Scissors!")

def detect_choice():
    detected_choice = None #player choice is none at first

    while detected_choice is None:#loop until something is detected
        ret, frame = cap.read()#get a frame from the camera

        if not ret:
            print("Failed to grab frame from camera")#if camera can't capture sad
            break

        #run yolov8 on the frame
        results = model(frame, verbose=False)#use verbose=False to hide details

        #get the objects detected in the frame
        boxes = results[0].boxes

        #clear the terminal  new output looks clean
        clear_terminal()

        
        for box in boxes:
            cls = int(box.cls[0])#(0=rock, 1=paper, 2=scissors)
            conf = box.conf[0]#confidence score 

            # if confidence is  >= 0.90 then it will do it unless no so i can avoid minor handmovments
            if conf >= 0.90:
                label = model.names[cls]#get the label of the detected object
                #checking if the detection is rock paper scissors
                if label.lower() in ["rock", "paper", "scissors"]:
                    detected_choice = label.capitalize()#store the players move
                    print(f"Detected: {detected_choice} with confidence: {conf:.2f}")
                    return detected_choice#return the move (rock, paper, scissors)

        #show the camera feed with detections on the screen
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Camera Detection", annotated_frame)

        #if i press q  exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #release the camera and close all windows when done
    cap.release()
    cv2.destroyAllWindows()

#main to control the whole system
def main():
    #countdown starting
    countdown()

    #detects the players choice using the camera
    player_choice = detect_choice()

    if player_choice:
                                #play the game and show the result
        result = play_rock_paper_scissors(player_choice)
        print(result)
    else:
        print("No valid choice detected. Please try again.")#if nothing is detected

if __name__ == "__main__":
    main()
