import cv2
import numpy as np
import requests
import discord
import io
from datetime import datetime, timedelta
from discord.ext import commands
from discord import Intents
from ultralytics import YOLO
import asyncio

from sort import Sort

# Initialize the Discord bot
TOKEN = ""  # Your Discord bot token
intents = Intents.default()

bot = commands.Bot(command_prefix="/", intents=intents)

# Discord webhook URL (keep it for scheduled screenshots)
webhook_url = 'https://discord.com/api/webhooks/1230800294506397716/wmsO_-f1mNLC6qVdLEpR8TIC_ixqp2Ynyt2J7pdxO_IcLsBuFAzLW1rkWMzqXNzTutUQ'

# Initialize YOLO and tracking variables
model = YOLO('/home/blueberry/moreai/dataset/runs/detect/train/weights/best.pt')
tracker = Sort()
memory = {}
line = [(100, 400), (500, 400)]  # Line to count cars
counter = 0  # Car counter
crossed_ids = set()  # Track unique car IDs

# Car timestamps for last 1 hour and 24 hours
car_timestamps = []

# Video Capture
vs = cv2.VideoCapture(0)  # Use camera
(W, H) = (None, None)

# Function to send images to the Discord webhook
async def send_to_discord(image_bytes, car_count):
    files = {'file': ('screenshot.jpg', image_bytes, 'image/jpeg')}
    payload = {
        "username": "Car Counter Bot",
        "content": f"Car count: {car_count}"
    }
    response = requests.post(webhook_url, data=payload, files=files)
    if response.status_code == 204:
        print("[INFO] Image successfully sent to Discord.")
    else:
        print(f"[ERROR] Failed to send image. Status Code: {response.status_code}")

# Helper function to capture and encode an image
def capture_image():
    grabbed, frame = vs.read()
    if not grabbed:
        return None
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_bytes = io.BytesIO(img_encoded.tobytes())
    return image_bytes

# Helper function to capture and send a video
async def capture_video():
    video_stream = io.BytesIO()
    video_writer = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_stream, video_writer, 20.0, (640, 480))

    # Capture frames for 5 seconds
    for _ in range(100):
        grabbed, frame = vs.read()
        if not grabbed:
            break
        out.write(frame)
        await asyncio.sleep(0.05)  # Simulate delay

    # Finish writing the video
    out.release()
    video_stream.seek(0)

    return video_stream

# Slash Commands using discord.py's built-in slash command support
@bot.tree.command(name="reset", description="Reset the car counter and send a snapshot")
async def reset(interaction: discord.Interaction):
    global counter, crossed_ids
    previous_count = counter
    counter = 0
    crossed_ids.clear()

    # Detect cars, update the frame with boxes and line
    frame = detect_and_count()

    # Encode the frame to an in-memory image file
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_bytes = io.BytesIO(img_encoded.tobytes())

    if image_bytes:
        await interaction.response.send_message(f"Car counter reset. Previous count: {previous_count}")
        await interaction.followup.send(file=discord.File(image_bytes, filename="reset_screenshot.jpg"))
    else:
        await interaction.response.send_message("Failed to capture image.")

# Slash command to send the current car count and a snapshot
@bot.tree.command(name="current", description="Send the current car count and a snapshot")
async def current(interaction: discord.Interaction):
    global counter

    # Detect cars, update the frame with boxes and line
    frame = detect_and_count()

    # Encode the frame to an in-memory image file
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_bytes = io.BytesIO(img_encoded.tobytes())

    if image_bytes:
        await interaction.response.send_message(f"Current car count: {counter}")
        await interaction.followup.send(file=discord.File(image_bytes, filename="current_screenshot.jpg"))
    else:
        await interaction.response.send_message("Failed to capture image.")

@bot.tree.command(name="video", description="Send a 5 second video")
async def video(interaction: discord.Interaction):
    # Capture and send the current video
    video_stream = await capture_video()
    if video_stream:
        await interaction.response.send_message("Here is the 5-second video", file=discord.File(video_stream, filename="video.mp4"))
    else:
        await interaction.response.send_message("Failed to capture video.")

# Function to detect cars and update the counter
# Function to detect cars and update the counter
# Function to detect cars, draw bounding boxes, and update the counter
def detect_and_count():
    global counter, crossed_ids, car_timestamps
    global W, H  # Declare W and H as global variables so they are accessible in this function

    grabbed, frame = vs.read()

    if not grabbed:
        return frame  # If no frame is grabbed, return an empty frame

    # If the frame dimensions are empty, grab them
    if W is None or H is None:
        H, W = frame.shape[:2]  # Set W and H based on the current frame

    # YOLO detection
    results = model(frame)
    boxes = []

    # Process YOLO results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0]  # Confidence of the detection
            classID = int(box.cls[0])  # Class ID of the detected object

            # Append bounding box coordinates to the list
            boxes.append([x1, y1, x2 - x1, y2 - y1])

            # Draw bounding box on the frame
            color = (0, 255, 0)  # Green for detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Car: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Prepare detections for SORT tracker
    if len(boxes) > 0:
        dets = np.array([[x, y, x + w, y + h, 1.0] for (x, y, w, h) in boxes])
    else:
        dets = np.empty((0, 5))  # If no detections, create an empty 2D array

    # Only update tracker if there are detections
    if dets.shape[0] > 0:
        tracks = tracker.update(dets)
    else:
        tracks = []

    # Draw the counting line (horizontal line example)
    cv2.line(frame, line[0], line[1], (0, 0, 255), 2)  # Red for the counting line

    # Check for tracked objects and if they crossed the counting line
    for track in tracks:
        track_id = int(track[4])
        if track_id not in crossed_ids:
            (x, y, w, h) = [int(v) for v in track[:4]]
            center_x = (x + w) // 2
            center_y = (y + h) // 2

            # Draw the center point of the object
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow for the center

            # Retrieve previous position of the car
            if track_id in memory:
                previous_box = memory[track_id]
                prev_center_y = (previous_box[1] + previous_box[3]) // 2

                # Check if the car moved from above to below the line (crossed the line)
                if prev_center_y < line[0][1] and center_y > line[0][1]:
                    counter += 1
                    crossed_ids.add(track_id)  # Mark this car as having crossed
                    car_timestamps.append(datetime.now())
                    
            # Store current position in memory for next iteration
            memory[track_id] = (x, y, w, h)

    # Return the frame with drawn detections and counting line
    return frame


# Background task to detect and count cars continuously
async def car_counter_task():
    while True:
        detect_and_count()
        await asyncio.sleep(0.1)  # Control the frequency of detection

# Run background task
@bot.event
async def on_ready():
    print(f"Bot {bot.user.name} is now running!")
    bot.loop.create_task(car_counter_task())
    await bot.tree.sync()  # Sync the commands with Discord

# Run the bot
bot.run(TOKEN)

