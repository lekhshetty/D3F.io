import os
import av
import pandas as pd
import threading
import streamlit as st
#import streamlit_nested_layout
import pymysql
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from pymysql.cursors import DictCursor
from audio_handling import AudioFrameHandler
from drowsy_detection import VideoFrameHandler
import plotly.express as px

# Add your database credentials here
db_credentials = {
    "host": "localhost",
    "user": "root",
    "password": "mysql",
    "database": "d3f",
}

def get_table_names():
    connection = pymysql.connect(**db_credentials, cursorclass=DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = "SHOW TABLES"
            cursor.execute(sql)
            tables = [row[f"Tables_in_{db_credentials['database']}"] for row in cursor.fetchall()]
    finally:
        connection.close()

    return tables

def load_data_from_table(table_name):
    connection = pymysql.connect(**db_credentials, cursorclass=DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = f"SELECT * FROM {table_name}"
            cursor.execute(sql)
            data = cursor.fetchall()
    finally:
        connection.close()

    return pd.DataFrame(data)

def create_dashboard(data):
    # Convert the timestamp to a pandas datetime object
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # EAR time series plot
    fig_ear = px.line(data, x='timestamp', y='EAR', title='Eye Aspect Ratio (EAR) over Time')
    st.plotly_chart(fig_ear)

    # MAR time series plot
    fig_mar = px.line(data, x='timestamp', y='MAR', title='Mouth Aspect Ratio (MAR) over Time')
    st.plotly_chart(fig_mar)

    # Eye shut counter and yawn counter
    st.subheader("Eye Shut & Yawn Counters")
    st.write(f"Eye Shut Counter: {data['eye_shut_counter'].max()}")
    st.write(f"Yawn Counter: {data['yawn_counter'].max()}")

    # Alarm count
    st.subheader("Alarm Count")
    alarm_count = data['alarm_on'].sum()
    st.write(f"Alarm triggered {alarm_count} times during the trip")


def create_table_and_insert_data(df):
    connection = pymysql.connect(**db_credentials, cursorclass=DictCursor)
    try:
        with connection.cursor() as cursor:
            # Create a unique table name
            table_name = f"trip_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            # Create the table
            sql_create = f"""
            CREATE TABLE {table_name} (
                timestamp DATETIME,
                EAR DOUBLE,
                MAR DOUBLE,
                eye_shut_counter INT,
                yawn_counter INT,
                alarm_on BOOLEAN
            )
            """
            cursor.execute(sql_create)
            connection.commit()

            # Insert data from the DataFrame into the table
            for _, row in df.iterrows():
                sql_insert = f"""
                INSERT INTO {table_name}
                (timestamp, EAR, MAR, eye_shut_counter, yawn_counter, alarm_on)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql_insert, row.to_list())
                connection.commit()
    finally:
        connection.close()


# Define the audio file to use.
path = os.path.dirname(__file__)
alarm_file_path = os.path.join(path,"audio", "wake_up.wav")

# Define pandas database that will be relayed to the backend MySQL database.
#drwsy_df=pd.DataFrame(columns=['timestamp', 'EAR', 'MAR', 'eye_shut_counter', 'yawn_counter', 'alarm_on'])
# if 'drwsy' not in st.session_state:
#         st.session_state['drwsy'] = pd.DataFrame(columns=['timestamp', 'EAR', 'MAR', 'eye_shut_counter', 'yawn_counter', 'alarm_on'])


# Streamlit Components
st.set_page_config(
    page_title="D3F",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Driver Drowsiness Detection and Feedback (D3F)",
    },
)
 
def main():
    st.title("D3F.io")
    st.subheader("Driver Drowsiness Detection and Feedback")

    prev_trip = st.button("View Previous Trips")
    
    if prev_trip:
        page2() 

    st.title("Drowsiness Detection")
    
    if 'drwsy' not in st.session_state:
        st.session_state['drwsy'] = pd.DataFrame(columns=['timestamp', 'EAR', 'MAR', 'eye_shut_counter', 'yawn_counter', 'alarm_on'])
    
    # col1, col2 = st.columns(spec=[1, 1])
    
    # with col1:
    #     # Lowest valid value of Eye Aspect Ratio. Ideal values [0.15, 0.2].
    #     EAR_THRESH = st.slider("Eye Aspect Ratio threshold:", 0.0, 0.4, 0.18, 0.01)
    
    # with col2:
    #     # The amount of time (in seconds) to wait before sounding the alarm.
    #     WAIT_TIME = st.slider("Seconds to wait before sounding alarm:", 0.0, 5.0, 1.0, 0.25)
    
    thresholds = {
        "EAR_THRESH": 0.18,
        "MAR_THRESH": 1.00,
        "WAIT_TIME": 4.0
    }
    
    # For streamlit-webrtc
    video_handler = VideoFrameHandler()
    audio_handler = AudioFrameHandler(sound_file_path=alarm_file_path)
    
    # For thread-safe access & to prevent race-condition.
    lock = threading.Lock()  
    
    #shared_state = {"play_alarm": False, "drwsy_df":pd.DataFrame(columns=['timestamp', 'EAR', 'MAR', 'eye_shut_counter', 'yawn_counter', 'alarm_on'])}
    shared_state = {"play_alarm": False}

    def video_frame_callback(frame: av.VideoFrame):
        frame = frame.to_ndarray(format="bgr24")  # Decode and convert frame to RGB
        #frame, play_alarm, row_dict = video_handler.process(frame, thresholds)  # Process frame
        frame, play_alarm = video_handler.process(frame, thresholds)  # Process frame
        with lock:
            shared_state["play_alarm"] = play_alarm  # Update shared state
            #shared_state['drwsy_df'] = pd.concat([shared_state['drwsy_df'], pd.DataFrame.from_records([row_dict])], ignore_index=True)
            #shared_state['drwsy_df'] = row_dict

        # Encode and return BGR frame
        return av.VideoFrame.from_ndarray(frame, format="bgr24")  
    
    def audio_frame_callback(frame: av.AudioFrame):
        with lock:  # access the current “play_alarm” state
            play_alarm = shared_state["play_alarm"]
    
        new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_alarm)
        return new_frame
    
    ctx = webrtc_streamer(
        key="driver-drowsiness-detection",
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        #media_stream_constraints={"video": {"width": True, "audio": True}},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False)
    )
    
    while ctx.state.playing:
        with lock:
            row_dict = video_handler.row_dict
        if row_dict is None:
            continue
        st.session_state['drwsy'] = pd.concat([st.session_state['drwsy'], pd.DataFrame.from_records([row_dict])], ignore_index=True)

    if st.button("End Trip"):
        # Send the data in the dataframe df to the MySQL database
        # create_table_and_insert_data(shared_state['drwsy_df'])
        create_table_and_insert_data(st.session_state['drwsy'])
        # page3(video_handler.df)
        # page3(video_handler.getdf())
        page2()
        

def page2():
    st.title("Trip Information")
    
    # Fetch table names from the database
    table_names = get_table_names()

    # Display the list of available tables
    selected_table = st.selectbox("Select a trip:", table_names)

    if selected_table:
        # Load data from the selected table
        trip_data = load_data_from_table(selected_table)

        # Create the dashboard with the data
        create_dashboard(trip_data)

    if st.button("Return Home"):
        main()

main()