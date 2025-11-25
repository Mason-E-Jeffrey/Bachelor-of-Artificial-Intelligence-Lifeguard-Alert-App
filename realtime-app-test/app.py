# Import required libraries for real-time video simulation, YOLOv8m inference, & sending SMS alerts.
from flask import Flask, request, render_template
import os
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, Point
from twilio.rest import Client
from ultralytics import YOLO
import supervision as sv
from supervision import Detections
import webbrowser

# Real-time performance settings for simulation. Note: Using < 1280 (training size) may cause the model to miss far-away swimmers.
image_size = 1280
frame_skip = 3

# SMS carrier domains & Twilio settings for sending the SMS text alert to lifeguards.
SMS_CARRIERS = {
    'Boost Mobile': '@smspremium.net.au', 
    'Telstra': '@mobilenet.telstra.com',
    'Optus': '@optusmobile.com.au',
    'Vodafone': '@vfx.ne.jp',
    'Virgin Mobile': '@vtext.com.au',
    'ALDI Mobile': '@mms.aldimobile.com.au'
}

MESSAGE_COOLDOWN = 60
LAST_MESSAGE_TIMES = {} 

# I've left our Twilio account details blank for privacy reasons.
TWILIO_ACCOUNT_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''
TWILIO_IS_TRIAL = True

# Holds the current alert message & when it should expire.
active_alert = {
  'message': '',
  'expires': 0
}

# Used for frame skipping to speed up detections & reduce lag.
previous_detections = None

# Setup Flask app as well as the upload folder for user inputs.
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utilize GPU for real-time YOLO-v8m inference. 
if torch.cuda.is_available():
  device = 'cuda'
  print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else: 
  device = 'cpu'
  print('Using CPU')

# Load the pre-trained best performing YOLO-m model & optimize the GPU.
model = YOLO('best.pt')
model.fuse()

if device == 'cuda':
    model.to(device=device)
    model.model.half()

torch.set_grad_enabled(False)

# There's only one 'person' class to detect, we segment them into three sub-groups based on which polygon zone they're in.
classes = [0] 

# Use ByteTrack for tracking people across frames & draw boxes & tracing lines to observe movement. 
byte_tracker = sv.ByteTrack(frame_rate=25)
box_annotator = sv.BoxAnnotator(thickness=2)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

# Send an SMS using Twilio. Note: you can still run real-time detection without a Twilio account, but this feature helps with crowd management.
def can_send_message(message_type):
  current_time = time.time()
  last_time = LAST_MESSAGE_TIMES.get(message_type, 0)
  if current_time - last_time >= MESSAGE_COOLDOWN:
    LAST_MESSAGE_TIMES[message_type] = current_time
    return True
  return False

def send_sms_alert(recipient_number, message, message_type='alert'):
  try:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
      print('Twilio account details not setup.')
      return

    if recipient_number.startswith('0'):
      recipient_number = '+61' + recipient_number[1:]
    elif not recipient_number.startswith('+'):
      recipient_number = '+' + recipient_number

    if not can_send_message(message_type):
      print(f'Throttled: Wait {MESSAGE_COOLDOWN}s between alerts.')
      return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    if TWILIO_IS_TRIAL:
      verified_numbers = client.outgoing_caller_ids.list()
      verified_numbers = [n.phone_number for n in verified_numbers]
      if recipient_number not in verified_numbers:
        print(f'Number {recipient_number} is not verified. Verify it at https://www.twilio.com/user/account/phone-numbers/verified')
        return

    msg = client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=recipient_number)
    print(f'SMS sent successfully: {msg.sid}')

  except Exception as e:
    print(f'SMS send error: {e}')

# Read both the water and flag zones from a single XML file.
def load_polygons_from_xml(xml_path):
  tree = ET.parse(xml_path)
  root = tree.getroot()
  water_poly, flags_poly = None, None
  for track in root.findall('.//track'):
    label = track.attrib.get('label')
    polygon_elem = track.find('polygon')
    if polygon_elem is None:
      continue
    raw_points = polygon_elem.attrib['points']
    points = [(float(x), float(y)) for x, y in (pt.split(',') for pt in raw_points.split(';'))]
    if label == 'water':
      water_poly = Polygon(points)
    elif label == 'flags':
      flags_poly = Polygon(points)
  return water_poly, flags_poly

# Use the center of a person's bounding box to classify them as in water, out of flags, or on beach 
def classify_person(x1, y1, x2, y2, water_poly, flags_poly):
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
  point = Point(cx, cy)
  if water_poly and water_poly.contains(point):
    if flags_poly and flags_poly.contains(point):
      return 'in_water'
    else:
      return 'out_of_flags'
  return 'on_beach'

# Plays the video, starts detection, & checks zones for each frame.
def fast_annotate_video(video_path, xml_path, options):
    video_cap = cv2.VideoCapture(video_path)
    water_poly, flags_poly = load_polygons_from_xml(xml_path)
    frame_index = 0

    while True:
        success, frame = video_cap.read()
        # Stop if no more frames
        if not success:
            break
        frame_index += 1
        # Perform detection & polygon zoning.
        processed_frame = process_frame(frame, water_poly, flags_poly, options, frame_index)
        # Display the annotated frame.
        cv2.imshow('Live', processed_frame)
        # Press 'q' to exit the real-time detection screen.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_cap.release()
    cv2.destroyAllWindows()
    
# Performs object detection & displays the zones on a frame.
def process_frame(frame, water_poly, flags_poly, options, frame_index=0):
    global active_alert, previous_detections

    # Run YOLOv8m every few frames to simulate real-time annotation.
    if frame_index % frame_skip == 0:
      results = model(
        frame,
        imgsz=image_size, # Match our training image size to retain important details (e.g. far-away swimmers).
        conf=0.4,     
        iou=0.5,
        classes=classes,
        max_det=150,    
        verbose=False
      )[0]

      detections = Detections.from_ultralytics(results)
      detections = detections[np.isin(detections.class_id, classes)]
      detections = byte_tracker.update_with_detections(detections)
      previous_detections = detections

    else:
      if previous_detections is None: 
        return frame

      detections = previous_detections

    # Track how many people are in each zone.
    in_water_count = 0
    out_of_flags_count = 0
    on_beach_count = 0

    labels = [
      f"{tracker_id} {model.names[class_id]} {confidence:.2f}"
      for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)

    for box, label_text in zip(detections.xyxy, labels):
      x1, y1, x2, y2 = box.astype(int)
      zone_label = classify_person(x1, y1, x2, y2, water_poly, flags_poly)

      if zone_label == 'in_water':
        in_water_count += 1
        colour = (0, 0, 255)     
      elif zone_label == 'out_of_flags':
        out_of_flags_count += 1
        colour = (255, 0, 0)     
      else:
        on_beach_count += 1
        colour = (0, 255, 255)   

      # Draw bounding boxes & labels.
      if (
        (zone_label == 'in_water' and options['show_in_water']) or
        (zone_label == 'out_of_flags' and options['show_out_of_flags']) or
        (zone_label == 'on_beach' and options['show_on_beach'])
        ):
      
        if options['show_boxes']:
          cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), colour, 2)

        if options['show_labels']:
          cv2.putText(annotated_frame, label_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cv2.putText(annotated_frame, zone_label, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    # Display the pre-defined zone boundaries.
    if water_poly:
      water_pts = np.array(water_poly.exterior.coords, dtype=np.int32)
      cv2.polylines(annotated_frame, [water_pts], isClosed=True, color=(255, 100, 100), thickness=2)
    if flags_poly:
      flag_pts = np.array(flags_poly.exterior.coords, dtype=np.int32)
      cv2.polylines(annotated_frame, [flag_pts], isClosed=True, color=(100, 255, 100), thickness=2)
    
    # Display counts on GUI.
    cv2.putText(annotated_frame, f"In Water: {in_water_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)       # Blue
    cv2.putText(annotated_frame, f"Out of Flags: {out_of_flags_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Red
    cv2.putText(annotated_frame, f"On Beach: {on_beach_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)     # Yellow

    now = time.time()
    # An alert can only be sent every 60s.
    cooldown = 60
    # The alert message stays on the screen for 5s.
    alert_duration = 5

    if now - options['last_alert_time'] > cooldown:
      if out_of_flags_count > options['max_out_of_flags']:
        alert_message = f'{out_of_flags_count} people OUTSIDE flags!'
        send_sms_alert(options['alert_phone'], alert_message)
        active_alert['message'] = alert_message
        active_alert['expires'] = now + alert_duration
        options['last_alert_time'] = now

      elif in_water_count > options['max_in_water']:
        alert_message = f'{in_water_count} people IN WATER above limit!'
        send_sms_alert(options['alert_phone'], alert_message)
        active_alert['message'] = alert_message
        active_alert['expires'] = now + alert_duration
        options['last_alert_time'] = now

    if active_alert['message'] and now < active_alert['expires']:
        cv2.putText(annotated_frame, active_alert['message'], (annotated_frame.shape[1] - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated_frame

# HTML upload page to allow the user to submit a video & custom XML file.
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        uploaded_video = request.files.get('video')
        uploaded_xml = request.files.get('xml')
        alert_phone = request.form.get('alert_phone')

        if not uploaded_video or not uploaded_xml:
            return "Please upload both a video file and an XML file.", 400

        video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.filename)
        xml_path = os.path.join(UPLOAD_FOLDER, uploaded_xml.filename)
        uploaded_video.save(video_path)
        uploaded_xml.save(xml_path)

        options = {
          'show_boxes': bool(request.form.get('show_boxes')),
          'show_labels': bool(request.form.get('show_labels')),
          'show_in_water': bool(request.form.get('show_in_water')),
          'show_out_of_flags': bool(request.form.get('show_out_of_flags')),
          'show_on_beach': bool(request.form.get('show_on_beach')),
          'alert_phone': alert_phone,
          'max_out_of_flags': int(request.form.get('max_out_of_flags')),
          'max_in_water': int(request.form.get('max_in_water')),
          'last_alert_time': 0
        }

        fast_annotate_video(video_path, xml_path, options)
        return f"""
          <h2>Real-time detection started!</h2>
          <p>Playing: {uploaded_video.filename} with {uploaded_xml.filename}</p>
          <p>A window should open automatically. Press <b>Q</b> to stop.</p>
          <form action="/" method="get">
            <button type="submit">Upload Another Video</button>
          </form>
        """
    return render_template('index.html')

# Start the app & open in browser.
if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=False, use_reloader=False)